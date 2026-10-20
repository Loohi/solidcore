//! Live B-rep access for `SmBrep::Point3DClassify`.
//!
//! Upstream: `source/SMLib/src/SmBrep.cpp` (`Point3DClassify`, `RayFire`),
//! `source/SMLib/src/SmBrepCache.cpp` (`BuildTrees`).
//!
//! | Rust | Upstream |
//! |------|----------|
//! | [`BrepPointObjects::boundary_intersection`] | `BrepPointSolve(INTERSECT, SINGLE)` and `LoadFromIntersection` |
//! | [`BrepPointObjects::regions`] | `SmBrep::GetRegions` |
//! | [`BrepPointObjects::ray_fire`] | `SmBrep::RayFire` |
//!
//! This is a data-model bridge: the B-rep, its identity, its cache slot and the
//! geometry owner mappings are borrowed from the caller. Geometry is never copied
//! or detached from its owner. The cache is built only when a solve needs it, with
//! the caller's kernel, and is reused within the caller's valid geometry state.
//! The caller must invalidate that slot when edits invalidate the spatial trees
//! (for example through `topology::brep::p2::notify` and `BrepCacheSlot`).
//! Classification handles may include the application's B-rep identity; the
//! default `EntityRef` handles are meaningful only within this one B-rep.
//!
//! Ray sectoring borrows the B-rep mutably because it may create UV trim curves.
//! `RayFire`'s returned intersection is translated with `try_map_objects`, retaining
//! every scalar and optional field. The coordinator never reads the ray's private
//! intersection scratch after an error, so the returned-value `RayFire` API suffices.
//! Unavailable face tolerance/trim queries remain errors of their real callees.
//! No new recursion, interior mutability, default kernel or synthetic geometry.

use super::BrepClassifyObjects;
use crate::geom::global_solver::{SolutionRequested, SolverOperation};
use crate::geom::object::ObjectRef;
use crate::intersect::point_class::{ClassObject, PointClassification};
use crate::intersect::topology_solver::p1::TopologySolver;
use crate::intersect::topology_solver::p2::TopologySolveOptions;
use crate::kernel::Kernel;
use crate::math::{Point3, Vec3, BIG_DOUBLE};
use crate::topology::brep::{p1, p2};
use crate::topology::brep_cache::{
    make_brep_cache, BrepCache, BrepObjects, CacheObject, CacheSource,
};
use crate::topology::{Brep, BrepId};
use crate::{EntityRef, Result, SolidError};

const OP: &str = "query::brep::objects::BrepPointObjects";
const ENTITY_IDENTITY: fn(EntityRef) -> Result<EntityRef> = Ok;
const GEOMETRY_AVAILABLE: fn() -> Result<()> = || Ok(());

/// The identity, owner mapping and spatial-cache slot of one live B-rep.
///
/// Keep this context paired with the B-rep it describes. Separating these borrows
/// lets a caller inspect or edit the B-rep between queries; edits that invalidate
/// the spatial trees must also invalidate `cache`, as for [`BrepPointObjects`].
pub struct BrepQueryContext<'a> {
    /// The caller's identity for this B-rep.
    pub brep_id: BrepId,
    /// Its cache slot, built lazily using the kernel supplied to the query.
    pub cache: &'a mut Option<BrepCache>,
    /// The exact object keys used by this B-rep's curve and surface owners.
    pub keys: &'a dyn Fn(CacheObject) -> ObjectRef,
    /// The caller's owner lookup for those keys.
    pub owner_of: &'a dyn Fn(ObjectRef) -> Option<ObjectRef>,
    geometry_check: &'a dyn Fn() -> Result<()>,
}

impl<'a> BrepQueryContext<'a> {
    /// Bind the caller-owned state without building a cache or reading geometry.
    #[must_use]
    pub fn new(
        brep_id: BrepId,
        cache: &'a mut Option<BrepCache>,
        keys: &'a dyn Fn(CacheObject) -> ObjectRef,
        owner_of: &'a dyn Fn(ObjectRef) -> Option<ObjectRef>,
    ) -> Self {
        Self {
            brep_id,
            cache,
            keys,
            owner_of,
            geometry_check: &GEOMETRY_AVAILABLE,
        }
    }

    /// Check that the caller's geometry is available before boundary or ray work.
    /// Region-only queries do not invoke this check. The default always succeeds.
    #[must_use]
    pub fn with_geometry_check(mut self, check: &'a dyn Fn() -> Result<()>) -> Self {
        self.geometry_check = check;
        self
    }

    /// Check the caller's representation before a query evaluates source geometry.
    ///
    /// Callers that sample a curve before classifying its point can use the same
    /// check as boundary and ray queries. Topology-only paths need not call it.
    pub fn check_geometry(&self) -> Result<()> {
        (self.geometry_check)()
    }

    /// Borrow the matching live B-rep for a query with scoped entity handles.
    #[must_use]
    pub fn objects<'query>(&'query mut self, brep: &'query mut Brep) -> BrepPointObjects<'query> {
        BrepPointObjects::new(
            brep,
            self.brep_id,
            &mut *self.cache,
            self.keys,
            self.owner_of,
        )
        .with_geometry_check(self.geometry_check)
    }
}

/// A mutable B-rep and its caller-owned query state.
///
/// `cache`, `keys` and `owner_of` must describe `brep`, as for [`BrepObjects`].
/// An already populated cache must still be valid for its geometry and context.
/// [`new`](Self::new) returns scoped `EntityRef` classifications;
/// [`with_handles`](Self::with_handles) preserves application-wide identities.
pub struct BrepPointObjects<'a, O = EntityRef> {
    brep: &'a mut Brep,
    brep_id: BrepId,
    cache: &'a mut Option<BrepCache>,
    keys: &'a dyn Fn(CacheObject) -> ObjectRef,
    owner_of: &'a dyn Fn(ObjectRef) -> Option<ObjectRef>,
    entity_handle: &'a dyn Fn(EntityRef) -> Result<O>,
    geometry_check: &'a dyn Fn() -> Result<()>,
}

impl<'a> BrepPointObjects<'a> {
    /// Borrow a B-rep for classifications whose entity ids are scoped to it.
    /// The supplied cache is left untouched until a boundary or ray solve needs it.
    #[must_use]
    pub fn new(
        brep: &'a mut Brep,
        brep_id: BrepId,
        cache: &'a mut Option<BrepCache>,
        keys: &'a dyn Fn(CacheObject) -> ObjectRef,
        owner_of: &'a dyn Fn(ObjectRef) -> Option<ObjectRef>,
    ) -> Self {
        Self::with_handles(brep, brep_id, cache, keys, owner_of, &ENTITY_IDENTITY)
    }
}

impl<'a, O: ClassObject> BrepPointObjects<'a, O> {
    /// Borrow a B-rep and translate each of its entities to a caller-owned handle.
    ///
    /// `entity_handle` must preserve the entity's kind and distinguish records in
    /// different B-reps. It is also used for regions; no arena index is interpreted
    /// as an application object key. Existing classification fields are not mapped
    /// into this B-rep, so retained fields may still name another caller object.
    #[must_use]
    pub fn with_handles(
        brep: &'a mut Brep,
        brep_id: BrepId,
        cache: &'a mut Option<BrepCache>,
        keys: &'a dyn Fn(CacheObject) -> ObjectRef,
        owner_of: &'a dyn Fn(ObjectRef) -> Option<ObjectRef>,
        entity_handle: &'a dyn Fn(EntityRef) -> Result<O>,
    ) -> Self {
        Self {
            brep,
            brep_id,
            cache,
            keys,
            owner_of,
            entity_handle,
            geometry_check: &GEOMETRY_AVAILABLE,
        }
    }

    /// Check the caller's geometric representation before accessing it.
    ///
    /// Called only for boundary and ray queries, before bounding boxes or cache
    /// construction; the single-region path without boundary tests bypasses it.
    /// The default always succeeds. A caller that cannot represent all native
    /// geometry can report that limitation explicitly instead of solving against
    /// substituted geometry. Such a caller may conservatively reject a ray whose
    /// locked box would otherwise provide an earlier ordinary failure.
    #[must_use]
    pub fn with_geometry_check(mut self, check: &'a dyn Fn() -> Result<()>) -> Self {
        self.geometry_check = check;
        self
    }

    fn entity_object(&self, entity: EntityRef) -> Result<O> {
        let object = (self.entity_handle)(entity)?;
        if object.class_kind() != entity.class_kind() {
            return Err(SolidError::invalid_input(
                OP,
                "entity handle mapping changed the topology kind",
            ));
        }
        Ok(object)
    }
}

fn cache_for<'cache>(
    kernel: &Kernel,
    brep: &Brep,
    slot: &'cache mut Option<BrepCache>,
    keys: &dyn Fn(CacheObject) -> ObjectRef,
) -> Result<&'cache BrepCache> {
    if slot.is_none() {
        *slot = Some(make_brep_cache(kernel, CacheSource::Brep(brep), keys)?);
    }
    slot.as_ref().ok_or_else(|| {
        SolidError::invalid_input(OP, "B-rep cache was not retained after construction")
    })
}

/// Read the topology object named by the boundary solver. Its edge/face solve
/// replaces geometry handles with the actual owner, so a remaining curve/surface
/// must not be silently relabelled as the record that happens to store it.
fn cache_entity(object: CacheObject) -> Result<EntityRef> {
    match object {
        CacheObject::Vertex(v) => Ok(EntityRef::Vertex(v)),
        CacheObject::Edge(e) => Ok(EntityRef::Edge(e)),
        CacheObject::Face(f) => Ok(EntityRef::Face(f)),
        _ => Err(SolidError::invalid_input(
            OP,
            "boundary hit is not a B-rep vertex, edge or face",
        )),
    }
}

impl<O: ClassObject> BrepClassifyObjects<O> for BrepPointObjects<'_, O> {
    fn boundary_intersection(
        &mut self,
        kernel: &Kernel,
        point: Point3,
        point_zone_tolerance: f64,
        classification: &mut PointClassification<O>,
    ) -> Result<bool> {
        (self.geometry_check)()?;
        let cache = cache_for(kernel, self.brep, self.cache, self.keys)?;
        let mut objects = BrepObjects {
            brep: self.brep,
            brep_id: self.brep_id,
            cache,
            keys: self.keys,
            owner_of: self.owner_of,
        };
        let solutions = TopologySolver::brep_point_solve(
            kernel,
            &mut objects,
            Some(CacheObject::Brep),
            point,
            TopologySolveOptions {
                operation: SolverOperation::Intersect,
                requested: SolutionRequested::Single,
                tolerance_3d: point_zone_tolerance,
                best_answer_so_far_sq: 0.0,
                opt_vectors: &[],
            },
        )?;
        let Some(solution) = solutions.as_slice().first() else {
            return Ok(false);
        };
        let object = solution
            .objects
            .first()
            .copied()
            .flatten()
            .ok_or_else(|| SolidError::invalid_input(OP, "null intersection object"))?;
        classification.load_from_intersection(
            point_zone_tolerance,
            self.entity_object(cache_entity(object)?)?,
            &solution.start,
            0,
            BIG_DOUBLE,
        )?;
        Ok(true)
    }

    fn regions(&self) -> Result<Vec<O>> {
        p1::regions(self.brep, None)?
            .into_iter()
            .map(|region| self.entity_object(EntityRef::Region(region)))
            .collect()
    }

    fn ray_fire(
        &mut self,
        kernel: &Kernel,
        point: Point3,
        direction: Vec3,
        point_zone_tolerance: f64,
        classification: &mut PointClassification<O>,
    ) -> Result<Option<O>> {
        (self.geometry_check)()?;
        let slot = &mut *self.cache;
        let keys = self.keys;
        let hit = p2::ray_fire_with_cache(
            kernel,
            self.brep,
            self.brep_id,
            move |brep| cache_for(kernel, brep, slot, keys),
            self.keys,
            self.owner_of,
            point,
            direction,
            point_zone_tolerance,
            None,
        )?;
        *classification = hit
            .intersection
            .try_map_objects(|entity| self.entity_object(entity))?;
        hit.region
            .map(|region| self.entity_object(EntityRef::Region(region)))
            .transpose()
    }
}
