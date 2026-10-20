//! B-rep queries and checked surface evaluation entry points.
//!
//! Upstream: `source/SM_API/src/SmApiSurfaces.cpp`,
//! `source/SMLib/inc/SmPolyDecimate.h`, `source/SMLib/src/SmPolyDecimate.cpp`,
//! `source/SMLib/inc/SmPoly.h`, `source/SMLib/src/SmBrep.cpp`.
//!
//! | Rust | Upstream |
//! |------|----------|
//! | [`evaluate_surface_point`] | `SmApiEvaluateSurfacePoint` |
//! | [`evaluate_surface_normal`] | `SmApiEvaluateSurfaceNormal` |
//! | [`evaluate_surface_derivatives`] | `SmApiEvaluateSurfaceDerivatives` |
//! | [`PolyVertexClass`] | `SmPolyVertexClass` |
//! | [`decimate_vertex_attribute`], [`DecimateVertexData`] | `SmDecimateVertexAttr` constructor/destructor |
//! | [`quadric_vertex_attribute`], [`QuadricVertexData`] | `SmQuadricVertexAttr` constructor/destructor |
//! | [`quadric_edge_attribute`], [`QuadricEdgeData`] | `SmQuadricEdgeAttr` constructor/destructor |
//! | [`crate::geom::attribute::Attribute::make_copy`] | the three `MakeCopy` overrides |
//! | [`point_3d_classify`], [`BrepClassifyObjects`] | `SmBrep::Point3DClassify` |
//! | [`calculate_bounding_box`], [`calculate_loose_bounding_box`], [`BoundingBoxOptions`] | `SmBrep::CalculateBoundingBox` |
//! | [`calculate_tight_bounding_box`] | `SmBrep::CalculateTightBoundingBox` |
//! | [`lock_bounding_box`] | `SmBrep::LockBoundingBox` |
//! | [`unlock_bounding_box`] | `SmBrep::UnLockBoundingBox` |
//!
//! Deviations: nullable surfaces are `Option<&Surface>`. Surface evaluations and
//! box calculations return their outputs, preserving failure atomicity where
//! upstream does. Classification retains its in/out state, including writes
//! before failure. The locked box is owned by the B-rep. Unfiltered object lists
//! use the native list order resolved by `topology::brep::p1`. Region ray firing
//! ([`BrepPointObjects`]) builds the B-rep cache before `RayFire`'s box instead of
//! inside the solve; both are ordinary failures that only select the next ray.
//! No input-driven recursion is added here.
//! Destructors use ownership rather than poisoning fields just before freeing.
//! The quadric edge constructor leaves its sort index uninitialised in C++; the
//! port represents that state by `None` instead of exposing undefined memory.

use crate::geom::extent3d::Extent3d;
use crate::geom::vector2d::Point2;
use crate::geom::vector3d::{Point3, Vec3};
use crate::kernel::Kernel;
use crate::nurbs::knots::Side;
use crate::surface::surface::Surface;
use crate::topology::{Brep, FaceId};
use crate::{Result, SolidError};

pub mod objects;
pub use objects::{BrepPointObjects, BrepQueryContext};

fn checked_surface<'a>(
    surface: Option<&'a Surface>,
    uv: Point2,
    op: &'static str,
) -> Result<&'a Surface> {
    let surface = surface.ok_or_else(|| SolidError::invalid_input(op, "null surface"))?;
    let domain = surface.natural_uv_domain();
    let lo = domain.min();
    let hi = domain.max();
    // Keep the negated conjunction: a NaN parameter must fail the API range test.
    if !(uv.x >= lo.x && uv.x <= hi.x && uv.y >= lo.y && uv.y <= hi.y) {
        return Err(SolidError::invalid_input(
            op,
            "parameter outside natural domain",
        ));
    }
    Ok(surface)
}

/// Evaluate a surface point after the API's inclusive natural-domain check.
///
/// # Errors
/// Returns invalid input for a null surface or an out-of-domain (including NaN)
/// parameter, otherwise propagates the surface evaluation error unchanged.
pub fn evaluate_surface_point(surface: Option<&Surface>, uv: Point2) -> Result<Point3> {
    checked_surface(surface, uv, "query::brep::evaluate_surface_point")?.evaluate_point(uv)
}

/// Evaluate the unit normal, with both derivative sides taken from the left.
///
/// # Errors
/// Those of [`evaluate_surface_point`] and the surface's normal evaluation.
pub fn evaluate_surface_normal(surface: Option<&Surface>, uv: Point2) -> Result<Vec3> {
    checked_surface(surface, uv, "query::brep::evaluate_surface_normal")?.evaluate_normal(
        uv,
        Side::Left,
        Side::Left,
    )
}

/// Evaluate `(point, du, dv)`, with both derivative sides taken from the left.
///
/// # Errors
/// Those of [`evaluate_surface_point`] and the surface's first derivatives.
pub fn evaluate_surface_derivatives(
    surface: Option<&Surface>,
    uv: Point2,
) -> Result<(Point3, Vec3, Vec3)> {
    checked_surface(surface, uv, "query::brep::evaluate_surface_derivatives")?
        .evaluate_1st_derivatives(uv, true, true)
}

/// The polygon vertex classification (`SmPolyVertexClass`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PolyVertexClass {
    /// `SM_PVC_NOT_COMPUTED`.
    NotComputed = 0,
    /// `SM_PVC_UNKNOWN`.
    Unknown = 1,
    /// `SM_PVC_SIMPLE`.
    Simple = 2,
    /// `SM_PVC_BOUNDARY`.
    Boundary = 3,
    /// `SM_PVC_INTERIOR`.
    Interior = 4,
    /// `SM_PVC_CORNER`.
    Corner = 5,
}

/// Decimation bookkeeping (`SmDecimateVertexAttr`).
#[derive(Debug, Clone, PartialEq)]
pub struct DecimateVertexData {
    /// `m_lSortIndex`; reset to zero when copied.
    pub sort_index: u64,
    /// `m_eVertClass`.
    pub vertex_class: PolyVertexClass,
    /// `m_dVertexDistance`.
    pub vertex_distance: f64,
    /// `m_dMaxFaceError`.
    pub max_face_error: f64,
    /// `m_vAveragePlaneNormal`.
    pub average_plane_normal: Vec3,
}

/// The quadratic error form (`SmQuadricVertexAttr`).
#[derive(Debug, Clone, PartialEq)]
pub struct QuadricVertexData {
    /// Three rows of `m_Q`.
    pub matrix: [Vec3; 3],
    /// `m_b`.
    pub linear: Vec3,
    /// `m_c`.
    pub constant: f64,
}

/// Edge contraction bookkeeping (`SmQuadricEdgeAttr`).
#[derive(Debug, Clone, PartialEq)]
pub struct QuadricEdgeData {
    /// `m_vNewPoint`.
    pub new_point: Point3,
    /// `m_dError`.
    pub error: f64,
    /// `m_lSortIndex`: unset until assigned; `MakeCopy` sets it to zero.
    pub sort_index: Option<u64>,
}

/// Construct `SmDecimateVertexAttr` (upstream default behaviour: `Copy`).
#[must_use]
pub fn decimate_vertex_attribute(
    id: u64,
    behavior: crate::geom::attribute::Behavior,
) -> crate::geom::attribute::Attribute {
    use crate::geom::attribute::{Attribute, AttributeValue};
    Attribute::with_value(
        id,
        behavior,
        AttributeValue::DecimateVertex(DecimateVertexData {
            sort_index: 0,
            vertex_class: PolyVertexClass::NotComputed,
            vertex_distance: -crate::math::BIG_DOUBLE,
            max_face_error: 0.0,
            average_plane_normal: Vec3::new(0.0, 0.0, 0.0),
        }),
    )
}

/// Construct `SmQuadricVertexAttr` (upstream default behaviour: `Copy`).
#[must_use]
pub fn quadric_vertex_attribute(
    id: u64,
    behavior: crate::geom::attribute::Behavior,
) -> crate::geom::attribute::Attribute {
    use crate::geom::attribute::{Attribute, AttributeValue};
    let zero = Vec3::new(0.0, 0.0, 0.0);
    Attribute::with_value(
        id,
        behavior,
        AttributeValue::QuadricVertex(QuadricVertexData {
            matrix: [zero; 3],
            linear: zero,
            constant: 0.0,
        }),
    )
}

/// Construct `SmQuadricEdgeAttr` (upstream default behaviour: `Copy`).
#[must_use]
pub fn quadric_edge_attribute(
    id: u64,
    behavior: crate::geom::attribute::Behavior,
) -> crate::geom::attribute::Attribute {
    use crate::geom::attribute::{Attribute, AttributeValue};
    Attribute::with_value(
        id,
        behavior,
        AttributeValue::QuadricEdge(QuadricEdgeData {
            new_point: Vec3::new(0.0, 0.0, 0.0),
            error: 0.0,
            sort_index: None,
        }),
    )
}

/// The boundary solve, region list and ray-fire access used by [`point_3d_classify`].
/// Handles remain caller-owned. [`BrepPointObjects`] resolves these operations
/// against a live B-rep and its spatial cache, preserving its concrete geometry.
/// Mutable access is required because ray sectoring may create UV trim curves.
pub trait BrepClassifyObjects<O> {
    /// The first `BrepPointSolve(INTERSECT, SINGLE)` boundary hit, loaded into
    /// `classification` with `LoadFromIntersection`. Returns false without writing
    /// it when there is no hit; errors retain any classification writes made.
    fn boundary_intersection(
        &mut self,
        kernel: &Kernel,
        point: Point3,
        point_zone_tolerance: f64,
        classification: &mut crate::intersect::point_class::PointClassification<O>,
    ) -> Result<bool>;
    /// Its complete region list in native list order (`GetRegions`).
    fn regions(&self) -> Result<Vec<O>>;
    /// `SmBrep::RayFire`: the containing region and the successful ray hit.
    /// The coordinator only consumes the region; the classification is private
    /// scratch, never part of its output. Ordinary failures select the next ray.
    fn ray_fire(
        &mut self,
        kernel: &Kernel,
        point: Point3,
        direction: Vec3,
        point_zone_tolerance: f64,
        classification: &mut crate::intersect::point_class::PointClassification<O>,
    ) -> Result<Option<O>>;
}

/// Classify a point against the B-rep boundary and its containing region.
/// `point_zone_tolerance` is an already constructed `SmZoneTol3d` value (see
/// [`crate::geom::tol::zone_tol3d`] for conversion from a raw tolerance).
///
/// The in/out classification preserves upstream's initial writes on failure and
/// retains its other fields. Boundary solving runs before the single-region
/// shortcut; region ray firing tries +X/-X, then +Y/-Y, then +Z/-Z only as needed.
///
/// # Errors
/// Those of boundary solving and classification loading. An unavailable callee
/// propagates `Unported`; ordinary ray-fire failures trigger the native retry
/// order, with a generic upstream error when no region is found.
pub fn point_3d_classify<O>(
    kernel: &Kernel,
    objects: &mut dyn BrepClassifyObjects<O>,
    point: Point3,
    point_zone_tolerance: f64,
    boundary_intersections: bool,
    classification: &mut crate::intersect::point_class::PointClassification<O>,
) -> Result<()>
where
    O: crate::intersect::point_class::ClassObject,
{
    use crate::error::ignored;
    use crate::intersect::point_class::{PointClassType, PointClassification};
    const OP: &str = "query::brep::point_3d_classify";
    classification.set_class_object(PointClassType::Unknown, None);
    classification.set_src_zone_tol_3d(point_zone_tolerance);
    if boundary_intersections
        && objects.boundary_intersection(kernel, point, point_zone_tolerance, classification)?
    {
        return Ok(());
    }
    let regions = objects.regions()?;
    if let [region] = regions.as_slice() {
        classification.set_class_object(PointClassType::Region, Some(*region));
        classification.set_gap_3d(0.0);
        classification.set_pre_snap_param(crate::math::BIG_DOUBLE);
        return Ok(());
    }
    let mut scratch = PointClassification::new(point_zone_tolerance);
    let mut region = None;
    for direction in [
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
    ] {
        region = match ignored(objects.ray_fire(
            kernel,
            point,
            direction,
            point_zone_tolerance,
            &mut scratch,
        ))? {
            Some(region) => region,
            None => ignored(objects.ray_fire(
                kernel,
                point,
                -direction,
                point_zone_tolerance,
                &mut scratch,
            ))?
            .flatten(),
        };
        if region.is_some() {
            break;
        }
    }
    let region = region.ok_or_else(|| SolidError::sm_err(OP))?;
    classification.set_class_object(PointClassType::Region, Some(region));
    classification.set_deviation(0.0); // SM_USE_OLDTOL; SetDeviation delegates SetGap3d.
    classification.set_pre_snap_param(crate::math::BIG_DOUBLE);
    Ok(())
}

/// Options for [`calculate_bounding_box`], matching the three native defaults.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundingBoxOptions {
    /// Skip faces; with more than 500 vertices AND more than 100 edges, also skip
    /// edges after including every vertex box (`bApprox`, default false).
    pub approximate: bool,
    /// Ask each edge and face for its tight box (`bTight`, default false).
    pub tight: bool,
    /// Expand each object's box by its own zone tolerance (default true).
    pub expand_by_zone_tolerance: bool,
}
impl Default for BoundingBoxOptions {
    fn default() -> Self {
        Self {
            approximate: false,
            tight: false,
            expand_by_zone_tolerance: true,
        }
    }
}
fn face_bounding_box(kernel: &Kernel, brep: &Brep, face: FaceId, out: &mut Extent3d) -> Result<()> {
    // The loose face calculation writes its caller's box only on success. In
    // particular, the tight B-rep loop reuses the previous face's box on failure.
    *out = crate::query::face::p1::calculate_bounding_box(kernel, brep, face, false)?;
    Ok(())
}
fn face_tight_bounding_box(
    kernel: &Kernel,
    brep: &Brep,
    face: FaceId,
    out: &mut Extent3d,
) -> Result<()> {
    // SmFace.cpp:1014 publishes the boundary box before fallible surface work;
    // the B-rep's tight calculation consumes that output even after an error.
    crate::query::face::p3::calculate_tight_bounding_box_into(kernel, brep, face, out)
}

/// The B-rep box, respecting a previously locked box before reading geometry.
/// Uses the complete native vertex, edge and face lists from
/// [`crate::topology::brep::p1`], in their recorded order. Tolerance expansion is
/// per object, in addition to any expansion performed by that object's box
/// calculation; disabling it does not disable the child's own expansion.
///
/// # Errors
/// Propagates the first failing list lookup or edge or face box calculation.
/// The result is returned only after the entire selected calculation succeeds.
pub fn calculate_bounding_box(
    kernel: &Kernel,
    brep: &Brep,
    options: BoundingBoxOptions,
) -> Result<Extent3d> {
    if !options.tight {
        return calculate_loose_bounding_box(
            kernel,
            brep,
            options.approximate,
            options.expand_by_zone_tolerance,
        );
    }
    let (mut bbox, faces) = bounding_box_before_faces(kernel, brep, options)?;
    for id in faces {
        let tolerance = brep.face(id)?.tolerance;
        let mut fbox = crate::query::face::p1::calculate_bounding_box(kernel, brep, id, true)?;
        if options.expand_by_zone_tolerance {
            fbox.expand_absolute(tolerance);
        }
        bbox = bbox.union(&fbox);
    }
    Ok(bbox)
}

/// The B-rep box with the native `bTight = false` behavior.
///
/// `approximate` and `expand_by_zone_tolerance` retain the corresponding options
/// of [`BoundingBoxOptions`]; their native defaults are `false` and `true`.
/// A previously locked box is returned before geometry is read.
///
/// # Errors
///
/// Those of the selected vertex, edge and face lists and box calculations.
pub fn calculate_loose_bounding_box(
    kernel: &Kernel,
    brep: &Brep,
    approximate: bool,
    expand_by_zone_tolerance: bool,
) -> Result<Extent3d> {
    let (mut bbox, faces) = bounding_box_before_faces(
        kernel,
        brep,
        BoundingBoxOptions {
            approximate,
            tight: false,
            expand_by_zone_tolerance,
        },
    )?;
    for id in faces {
        let tolerance = brep.face(id)?.tolerance;
        let mut fbox = crate::query::face::p1::calculate_bounding_box(kernel, brep, id, false)?;
        if expand_by_zone_tolerance {
            fbox.expand_absolute(tolerance);
        }
        bbox = bbox.union(&fbox);
    }
    Ok(bbox)
}

/// The common lock, vertex and edge stages. No faces remain after a locked-box or
/// approximate return; otherwise their recorded order is passed to the caller.
fn bounding_box_before_faces(
    kernel: &Kernel,
    brep: &Brep,
    options: BoundingBoxOptions,
) -> Result<(Extent3d, Vec<FaceId>)> {
    use crate::nurbs::b_spline_curve::BoundingBoxRequest;
    use crate::topology::brep::p1;
    if let Some(locked) = brep.locked_bounding_box {
        return Ok((locked, Vec::new()));
    }
    let mut bbox = Extent3d::new();
    let vertices = p1::vertices(brep, None)?;
    for &id in &vertices {
        let vertex = brep.vertex(id)?;
        let mut vbox = Extent3d::from_point(vertex.point);
        if options.expand_by_zone_tolerance {
            vbox.expand_absolute(vertex.tolerance);
        }
        bbox = bbox.union(&vbox);
    }
    let edges = p1::edges(brep, None)?;
    if options.approximate && vertices.len() > 500 && edges.len() > 100 {
        return Ok((bbox, Vec::new()));
    }
    let request = BoundingBoxRequest {
        normal: true,
        pseudo: false,
        polar: false,
    };
    for id in edges {
        let edge = brep.edge(id)?;
        if edge.curve.is_some() {
            let mut ebox = crate::query::edge::calculate_bounding_box(
                kernel,
                brep,
                id,
                request,
                options.tight,
            )?
            .normal
            .ok_or_else(|| SolidError::sm_err("query::brep::calculate_bounding_box"))?;
            if options.expand_by_zone_tolerance {
                ebox.expand_absolute(edge.tolerance);
            }
            bbox = bbox.union(&ebox);
        }
    }
    let faces = if options.approximate {
        Vec::new()
    } else {
        p1::faces(brep, None)?
    };
    Ok((bbox, faces))
}

/// The tight B-rep box; adds no zone tolerances and ignores the locked box.
/// Visits the complete native vertex, edge and face lists in their recorded order.
/// Loose boxes contained by the accumulated box skip the expensive tight call.
/// As upstream, ordinary edge/face calculation failures are ignored, retaining
/// their defined output state; missing callees still propagate `Unported`.
///
/// # Errors
/// An unavailable face/edge callee or an invalid arena reference.
pub fn calculate_tight_bounding_box(kernel: &Kernel, brep: &Brep) -> Result<Extent3d> {
    use crate::error::ignored;
    use crate::nurbs::b_spline_curve::BoundingBoxRequest;
    use crate::topology::brep::p1;
    let mut bbox = Extent3d::new();
    for id in p1::vertices(brep, None)? {
        let vbox = Extent3d::from_point(brep.vertex(id)?.point);
        bbox = bbox.union(&vbox);
    }
    let request = BoundingBoxRequest {
        normal: true,
        pseudo: false,
        polar: false,
    };
    for id in p1::edges(brep, None)? {
        brep.edge(id)?;
        // SmEdge::CalculateBoundingBox always initialises the normal box first.
        // The normal-only curve routines write their final box after fallible
        // work; a composite ignores child statuses and returns its union. Thus
        // an ordinary error leaves this freshly initialised empty edge box.
        let mut ebox = match ignored(crate::query::edge::calculate_bounding_box(
            kernel, brep, id, request, false,
        ))? {
            Some(boxes) => boxes
                .normal
                .ok_or_else(|| SolidError::sm_err("query::brep::calculate_tight_bounding_box"))?,
            None => Extent3d::new(),
        };
        if ebox.is_contained_by(&bbox, 0.0) {
            continue;
        }
        // CalculateTightBoundingBox writes only on success; without a curve it
        // succeeds without touching the caller's existing loose box.
        if let Some(Some(tight)) = ignored(crate::query::edge::calculate_tight_bounding_box(
            kernel, brep, id,
        ))? {
            ebox = tight;
        }
        bbox = bbox.union(&ebox);
    }
    let mut face_box = Extent3d::new();
    for id in p1::faces(brep, None)? {
        brep.face(id)?;
        crate::error::ignore_status(face_bounding_box(kernel, brep, id, &mut face_box))?;
        if face_box.is_contained_by(&bbox, 0.0) {
            continue;
        }
        crate::error::ignore_status(face_tight_bounding_box(kernel, brep, id, &mut face_box))?;
        bbox = bbox.union(&face_box);
    }
    Ok(bbox)
}

/// Replace the locked box, computing a tight box when requested. The previous
/// lock is released BEFORE calculation, and stays released if calculation fails.
///
/// # Errors
/// Those of the selected box calculation.
pub fn lock_bounding_box(kernel: &Kernel, brep: &mut Brep, tight: bool) -> Result<()> {
    unlock_bounding_box(brep);
    let bbox = if tight {
        calculate_tight_bounding_box(kernel, brep)?
    } else {
        calculate_bounding_box(kernel, brep, BoundingBoxOptions::default())?
    };
    brep.locked_bounding_box = Some(bbox);
    Ok(())
}

/// Release the locked box, also succeeding when no box is locked.
pub fn unlock_bounding_box(brep: &mut Brep) {
    brep.locked_bounding_box = None;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::EntityRef;
    use crate::geom::attribute::{AttributeValue, Behavior};
    use crate::intersect::point_class::{PointClassType, PointClassification};

    #[test]
    fn copied_brep_does_not_reuse_the_original_locked_box() {
        let k = Kernel::default();
        let mut original = Brep::new();
        let vertex = original.add_vertex(Vec3::ZERO).unwrap();
        lock_bounding_box(&k, &mut original, false).unwrap();
        let original_box = calculate_loose_bounding_box(&k, &original, false, true).unwrap();

        // SmBrep's copy constructor sets m_pLockedBoundingBox to NULL.
        let mut copied = original.clone();
        assert!(copied.locked_bounding_box.is_none());
        copied.vertex_mut(vertex).unwrap().point = Vec3::new(10.0, 20.0, 30.0);
        let copied_box = calculate_loose_bounding_box(&k, &copied, false, true).unwrap();
        assert!(copied_box.min().x > original_box.max().x);

        // The original keeps returning its locked box after its vertex moves.
        original.vertex_mut(vertex).unwrap().point = Vec3::new(-5.0, 0.0, 0.0);
        assert_eq!(
            calculate_bounding_box(&k, &original, BoundingBoxOptions::default()).unwrap(),
            original_box
        );
        unlock_bounding_box(&mut original);
        unlock_bounding_box(&mut original);
        let unlocked =
            calculate_bounding_box(&k, &original, BoundingBoxOptions::default()).unwrap();
        assert!(unlocked.min().x < original_box.min().x);
    }

    #[test]
    fn tight_box_ignores_lock_and_zone_tolerances() {
        let k = Kernel::default();
        let mut b = Brep::new();
        b.add_vertex(Vec3::new(1.0, 2.0, 3.0)).unwrap();
        b.add_vertex(Vec3::new(-1.0, 0.0, 4.0)).unwrap();
        let tight = calculate_tight_bounding_box(&k, &b).unwrap();
        assert_eq!(tight.min(), Vec3::new(-1.0, 0.0, 3.0));
        assert_eq!(tight.max(), Vec3::new(1.0, 2.0, 4.0));
        let loose = calculate_bounding_box(&k, &b, BoundingBoxOptions::default()).unwrap();
        assert!(loose.min().x < tight.min().x && loose.max().z > tight.max().z);
        let unexpanded = calculate_bounding_box(
            &k,
            &b,
            BoundingBoxOptions {
                expand_by_zone_tolerance: false,
                ..BoundingBoxOptions::default()
            },
        )
        .unwrap();
        assert_eq!(unexpanded, tight);
        lock_bounding_box(&k, &mut b, true).unwrap();
        assert_eq!(b.locked_bounding_box, Some(tight));
    }

    #[test]
    fn decimation_copies_reset_sort_indices() {
        let vertex = decimate_vertex_attribute(7, Behavior::Copy);
        let edge = quadric_edge_attribute(8, Behavior::Copy);
        let quadric = quadric_vertex_attribute(9, Behavior::Copy);
        assert!(matches!(
            edge.value(),
            AttributeValue::QuadricEdge(QuadricEdgeData {
                sort_index: None,
                ..
            })
        ));
        let mut vertex = vertex;
        if let AttributeValue::DecimateVertex(data) = vertex.value_mut() {
            data.sort_index = 41;
        }
        assert!(matches!(
            vertex.make_copy().value(),
            AttributeValue::DecimateVertex(DecimateVertexData { sort_index: 0, .. })
        ));
        assert!(matches!(
            edge.make_copy().value(),
            AttributeValue::QuadricEdge(QuadricEdgeData {
                sort_index: Some(0),
                ..
            })
        ));
        assert_eq!(quadric.make_copy(), quadric);
    }

    struct Fixed {
        regions: Vec<EntityRef>,
        rays: Vec<Result<Option<EntityRef>>>,
        fired: Vec<Vec3>,
    }

    impl BrepClassifyObjects<EntityRef> for Fixed {
        fn boundary_intersection(
            &mut self,
            _kernel: &Kernel,
            _point: Point3,
            _point_zone_tolerance: f64,
            _classification: &mut PointClassification<EntityRef>,
        ) -> Result<bool> {
            Ok(false)
        }
        fn regions(&self) -> Result<Vec<EntityRef>> {
            Ok(self.regions.clone())
        }
        fn ray_fire(
            &mut self,
            _kernel: &Kernel,
            _point: Point3,
            direction: Vec3,
            _point_zone_tolerance: f64,
            _classification: &mut PointClassification<EntityRef>,
        ) -> Result<Option<EntityRef>> {
            self.fired.push(direction);
            if self.rays.is_empty() {
                Ok(None)
            } else {
                self.rays.remove(0)
            }
        }
    }

    #[test]
    fn classify_retries_rays_in_native_order() {
        use crate::topology::RegionId;
        let k = Kernel::default();
        let r = |i| EntityRef::Region(RegionId::from_index(i));
        // A single region is returned without firing rays.
        let mut single = Fixed {
            regions: vec![r(0)],
            rays: Vec::new(),
            fired: Vec::new(),
        };
        let mut pc = PointClassification::new(1e-6);
        point_3d_classify(&k, &mut single, Vec3::ZERO, 1e-6, true, &mut pc).unwrap();
        assert_eq!(pc.point_class(), PointClassType::Region);
        assert_eq!(pc.object(), Some(r(0)));
        assert!(single.fired.is_empty());

        // An ordinary failure and an empty hit move on: +X, -X, then +Y.
        let mut several = Fixed {
            regions: vec![r(0), r(1)],
            rays: vec![Err(SolidError::sm_err("ray")), Ok(None), Ok(Some(r(1)))],
            fired: Vec::new(),
        };
        point_3d_classify(&k, &mut several, Vec3::ZERO, 1e-6, false, &mut pc).unwrap();
        assert_eq!(pc.object(), Some(r(1)));
        assert_eq!(
            several.fired,
            [
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(-1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0)
            ]
        );

        // A successful ray without a region skips the opposite direction, as
        // upstream only retries on a failed status; no region at the end is SM_ERR.
        // Unported propagates at once.
        let mut none = Fixed {
            regions: vec![r(0), r(1)],
            rays: Vec::new(),
            fired: Vec::new(),
        };
        assert!(matches!(
            point_3d_classify(&k, &mut none, Vec3::ZERO, 1e-6, false, &mut pc),
            Err(SolidError::Upstream { .. })
        ));
        assert_eq!(
            none.fired,
            [
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0)
            ]
        );
        let mut unported = Fixed {
            regions: vec![r(0), r(1)],
            rays: vec![Err(SolidError::unported("x", "y"))],
            fired: Vec::new(),
        };
        assert!(matches!(
            point_3d_classify(&k, &mut unported, Vec3::ZERO, 1e-6, false, &mut pc),
            Err(SolidError::Unported { .. })
        ));
        assert_eq!(unported.fired.len(), 1);
    }
}
