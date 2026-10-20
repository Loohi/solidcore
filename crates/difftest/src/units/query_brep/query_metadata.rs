//! Native query metadata for the shared real B-rep graph importers.
//!
//! Paired with `harness/cpp/shim/sc_brep_query_metadata.h`. Malformed metadata is
//! an error. An exact native state that the importer cannot represent is retained
//! in a report instead: known lists and owners are installed without replacing
//! geometry or silently following an indirect owner chain. Callers check topology
//! limitations before topology work, and geometric limitations only when that
//! geometry is consumed (for example via `BrepQueryContext::with_geometry_check`).

use std::collections::{BTreeMap, BTreeSet};

use solidcore::geom::a_object::BREP_TYPE;
use solidcore::geom::object::ObjectRef;
use solidcore::topology::brep_cache::CacheObject;
use solidcore::topology::{Brep, EdgeId, FaceId, FaceuseId, RegionId, VertexId};

use crate::{HarnessError, Output};

type Result<T> = std::result::Result<T, HarnessError>;

const EXPORT_LIMITATION: &str = "harness:analytic-geometry-export";

/// Representation limitations found while importing a real native B-rep.
///
/// Check topology with `deferred_output_for(false, false)` before using imported
/// lists. Curve and surface limitations can be checked later, when the operation
/// actually reads those kinds of geometry.
#[derive(Clone, Debug, Default)]
pub(crate) struct QueryMetadata {
    topology: Vec<String>,
    curves: BTreeMap<EdgeId, Vec<String>>,
    surfaces: Vec<String>,
}

impl QueryMetadata {
    /// Explain only limitations relevant to the requested geometric reads.
    ///
    /// Topology is always included. In particular, a vertices-only or approximate
    /// bounding-box path need not reject an unrepresented surface or curve owner.
    pub(crate) fn deferred_output_for(
        &self,
        read_curves: bool,
        read_surfaces: bool,
    ) -> Option<Output> {
        let mut reasons: Vec<&str> = self.topology.iter().map(String::as_str).collect();
        if read_curves {
            reasons.extend(self.curves.values().flatten().map(String::as_str));
        }
        if read_surfaces {
            reasons.extend(self.surfaces.iter().map(String::as_str));
        }
        (!reasons.is_empty()).then(|| Output::deferred(EXPORT_LIMITATION, reasons.join("; ")))
    }
}

fn error(message: impl Into<String>) -> HarnessError {
    HarnessError::new(format!("B-rep query metadata: {}", message.into()))
}

enum NativeValue<T> {
    Known(T),
    Unrepresented,
}

struct Reader<'a> {
    raw: &'a [f64],
    at: usize,
}

impl Reader<'_> {
    fn scalar(&mut self, name: &str) -> Result<f64> {
        let value = *self
            .raw
            .get(self.at)
            .ok_or_else(|| error(format!("missing {name}")))?;
        self.at += 1;
        if !value.is_finite() {
            return Err(error(format!("non-finite {name}")));
        }
        Ok(value)
    }

    fn integer(&mut self, name: &str) -> Result<i64> {
        let value = self.scalar(name)?;
        // Metadata integers are indices or native types/flags. Keep conversion
        // inside f64's exact integer range as well as i64's range.
        if value.fract() != 0.0 || value.abs() > (1_u64 << 53) as f64 {
            return Err(error(format!("invalid integer {name}")));
        }
        Ok(value as i64)
    }

    fn count(&mut self, name: &str) -> Result<usize> {
        let count =
            usize::try_from(self.integer(name)?).map_err(|_| error(format!("negative {name}")))?;
        // Bound work and allocations by the remaining input before iterating.
        if count > self.raw.len().saturating_sub(self.at) {
            return Err(error(format!("truncated {name}")));
        }
        Ok(count)
    }

    fn exact_count(&mut self, expected: usize, name: &str) -> Result<()> {
        if self.count(name)? != expected {
            return Err(error(format!("{name} differs from the imported graph")));
        }
        Ok(())
    }

    fn registry_id(
        &mut self,
        maximum: usize,
        name: &str,
        limitations: &mut Vec<String>,
    ) -> Result<NativeValue<Option<u32>>> {
        match self.integer(name)? {
            -1 => Ok(NativeValue::Known(None)),
            -3 => {
                limitations.push(format!("{name} is outside the exported topology registry"));
                Ok(NativeValue::Unrepresented)
            }
            id if id >= 0 && (id as u64) < maximum as u64 => {
                let id = u32::try_from(id).map_err(|_| error(format!("{name} id overflow")))?;
                Ok(NativeValue::Known(Some(id)))
            }
            _ => Err(error(format!("invalid {name} registry id"))),
        }
    }

    fn list(
        &mut self,
        maximum: usize,
        full: bool,
        name: &str,
        limitations: &mut Vec<String>,
    ) -> Result<NativeValue<Vec<u32>>> {
        let count = self.count(name)?;
        let mut represented = true;
        if full && count != maximum {
            limitations.push(format!(
                "{name} contains {count} of {maximum} imported records; the current B-rep model requires a full list"
            ));
            represented = false;
        }
        let mut list = Vec::new();
        let mut seen = BTreeSet::new();
        for _ in 0..count {
            match self.registry_id(maximum, name, limitations)? {
                NativeValue::Known(Some(id)) => {
                    if !seen.insert(id) {
                        return Err(error(format!("duplicate id in {name}")));
                    }
                    list.push(id);
                }
                NativeValue::Known(None) => return Err(error(format!("null member in {name}"))),
                NativeValue::Unrepresented => represented = false,
            }
        }
        Ok(if represented {
            NativeValue::Known(list)
        } else {
            NativeValue::Unrepresented
        })
    }

    fn geometry(
        &mut self,
        imported_type: i64,
        expected_owner: (i64, usize),
        owner_key: impl Fn(u32) -> ObjectRef,
        name: &str,
        limitations: &mut Vec<String>,
    ) -> Result<NativeValue<Option<ObjectRef>>> {
        let (owner_type, owner_count) = expected_owner;
        let native_type = self.integer("concrete geometry type")?;
        let native_owner_type = self.integer("direct owner type")?;
        let owner_id = self.integer("direct owner id")?;
        if native_type < -1 || native_owner_type < 0 || !matches!(owner_id, -3 | -1 | 0..) {
            return Err(error(format!("invalid {name} geometry record")));
        }
        if native_type != imported_type {
            limitations.push(format!(
                "{name} concrete type {native_type} is represented as {imported_type}; exact geometry is required"
            ));
        }
        if native_owner_type == 0 && owner_id == -1 {
            return Ok(NativeValue::Known(None));
        }
        if native_type == -1 || native_owner_type == 0 || owner_id == -1 {
            return Err(error(format!("inconsistent {name} owner record")));
        }
        if native_owner_type != owner_type || owner_id == -3 {
            limitations.push(format!(
                "{name} has an unrepresented direct owner (type {native_owner_type}, registry id {owner_id}); no owner-chain substitution is allowed"
            ));
            return Ok(NativeValue::Unrepresented);
        }
        let id = usize::try_from(owner_id).map_err(|_| error("owner id overflow"))?;
        if id >= owner_count {
            return Err(error(format!("{name} owner is outside the imported graph")));
        }
        let key = owner_key(u32::try_from(id).map_err(|_| error("owner id overflow"))?);
        if key.type_code != owner_type {
            return Err(error(format!("{name} owner-key mapping changes its type")));
        }
        Ok(NativeValue::Known(Some(key)))
    }
}

/// Install the representable metadata of the same native B-rep as brep.
///
/// keys supplies the caller's exact B-rep, face and edge identities. Parsing and
/// validation finish before writes begin. Concrete geometry is never replaced;
/// unknown owners and unrepresentable lists are left unchanged and reported.
/// Existing caches must be absent or invalidated before this fixture setup.
#[allow(clippy::too_many_lines)]
pub(crate) fn apply_query_metadata(
    brep: &mut Brep,
    raw: &[f64],
    keys: &dyn Fn(CacheObject) -> ObjectRef,
) -> Result<QueryMetadata> {
    let mut reader = Reader { raw, at: 0 };
    let mut report = QueryMetadata::default();
    if reader.integer("version")? != 1 {
        return Err(error("unsupported version"));
    }
    let tolerance = reader.scalar("B-rep tolerance")?;
    let infinite = reader.registry_id(
        brep.regions().len(),
        "infinite region",
        &mut report.topology,
    )?;
    reader.exact_count(brep.regions().len(), "region count")?;
    let mut void_flags = Vec::new();
    for _ in brep.regions().ids() {
        void_flags.push(
            i32::try_from(reader.integer("region void flag")?)
                .map_err(|_| error("region void flag overflow"))?,
        );
    }
    let regions = reader.list(
        brep.regions().len(),
        false,
        "region list",
        &mut report.topology,
    )?;
    let faces = reader.list(brep.faces().len(), true, "face list", &mut report.topology)?;
    let edges = reader.list(brep.edges().len(), true, "edge list", &mut report.topology)?;
    let vertices = reader.list(
        brep.vertices().len(),
        true,
        "vertex list",
        &mut report.topology,
    )?;

    reader.exact_count(brep.edges().len(), "curve owner count")?;
    let mut curve_owners = Vec::new();
    for (id, edge) in brep.edges().iter() {
        let mut limitations = Vec::new();
        curve_owners.push(
            reader.geometry(
                edge.curve
                    .as_ref()
                    .map_or(-1, solidcore::curve::curve::Curve::type_code),
                (solidcore::topology::edge::TYPE, brep.edges().len()),
                |id| keys(CacheObject::Edge(EdgeId::from_index(id))),
                &format!("edge {} curve", id.index()),
                &mut limitations,
            )?,
        );
        if !limitations.is_empty() {
            report.curves.insert(id, limitations);
        }
    }
    reader.exact_count(brep.faces().len(), "surface owner count")?;
    let mut surface_owners = Vec::new();
    for (id, face) in brep.faces().iter() {
        surface_owners.push(
            reader.geometry(
                face.surface
                    .as_ref()
                    .map_or(-1, solidcore::surface::surface::Surface::type_code),
                (solidcore::topology::face::p2::TYPE, brep.faces().len()),
                |id| keys(CacheObject::Face(FaceId::from_index(id))),
                &format!("face {} surface", id.index()),
                &mut report.surfaces,
            )?,
        );
    }
    reader.exact_count(brep.faces().len(), "face state count")?;
    let mut face_states = Vec::new();
    for id in brep.faces().ids() {
        let faceuse = reader.registry_id(
            brep.faceuses().len(),
            &format!("face {} primary faceuse", id.index()),
            &mut report.topology,
        )?;
        let rectangular_trim = i32::try_from(reader.integer("rectangular-trim flag")?)
            .map_err(|_| error("rectangular-trim flag overflow"))?;
        face_states.push((faceuse, rectangular_trim));
    }
    if reader.at != raw.len() {
        return Err(error("trailing values"));
    }
    let brep_key = keys(CacheObject::Brep);
    if brep_key.type_code != BREP_TYPE {
        return Err(error("B-rep key mapping changes its type"));
    }

    brep.set_key(brep_key.key);
    brep.set_tolerance(tolerance);
    if let NativeValue::Known(infinite) = infinite {
        brep.set_infinite_region(infinite.map(RegionId::from_index));
    }
    let region_ids: Vec<_> = brep.regions().ids().collect();
    for (id, is_void) in region_ids.into_iter().zip(void_flags) {
        let region = brep.region_mut(id).map_err(|e| error(e.to_string()))?;
        region.is_void = is_void;
    }
    if let NativeValue::Known(list) = regions {
        // Main's B-rep keeps its region list in arena order, rotated by the list
        // head (`topology::brep::p1::regions`). A native order that is not such a
        // rotation is a representation limitation, not something to approximate.
        let list: Vec<RegionId> = list.into_iter().map(RegionId::from_index).collect();
        let on_list: Vec<RegionId> = brep
            .regions()
            .iter()
            .filter(|(_, r)| r.in_brep)
            .map(|(id, _)| id)
            .collect();
        let start = list
            .first()
            .and_then(|head| on_list.iter().position(|r| r == head));
        let rotation = start.map(|start| {
            on_list
                .iter()
                .skip(start)
                .chain(on_list.iter().take(start))
                .copied()
                .collect::<Vec<_>>()
        });
        if list.is_empty() && on_list.is_empty() {
        } else if rotation.as_deref() == Some(list.as_slice()) {
            brep.set_region_list_head(list.first().copied());
        } else {
            report
                .topology
                .push("native region list order is not an arena-order rotation".to_owned());
        }
    }
    if let NativeValue::Known(list) = faces {
        brep.set_face_list(list.into_iter().map(FaceId::from_index).collect())
            .map_err(|e| error(e.to_string()))?;
    }
    if let NativeValue::Known(list) = edges {
        brep.set_edge_list(list.into_iter().map(EdgeId::from_index).collect())
            .map_err(|e| error(e.to_string()))?;
    }
    if let NativeValue::Known(list) = vertices {
        brep.set_vertex_list(list.into_iter().map(VertexId::from_index).collect())
            .map_err(|e| error(e.to_string()))?;
    }
    let edge_ids: Vec<_> = brep.edges().ids().collect();
    for (id, owner) in edge_ids.into_iter().zip(curve_owners) {
        if let NativeValue::Known(owner) = owner {
            if let Some(curve) = brep
                .edge_mut(id)
                .map_err(|e| error(e.to_string()))?
                .curve
                .as_mut()
            {
                let _ = curve.set_owner(owner, owner.map(|_| brep_key));
            }
        }
    }
    let face_ids: Vec<_> = brep.faces().ids().collect();
    for ((id, owner), (faceuse, rectangular_trim)) in
        face_ids.into_iter().zip(surface_owners).zip(face_states)
    {
        let face = brep.face_mut(id).map_err(|e| error(e.to_string()))?;
        if let NativeValue::Known(owner) = owner {
            if let Some(surface) = face.surface.as_mut() {
                surface.set_owner(owner);
            }
        }
        if let NativeValue::Known(faceuse) = faceuse {
            face.faceuse = faceuse.map(FaceuseId::from_index);
        }
        face.rectangular_trim = rectangular_trim;
    }
    Ok(report)
}

#[cfg(test)]
mod curve_guard_tests {
    use super::*;
    use solidcore::curve::curve::Curve;
    use solidcore::math::{Point3, Vec3};
    use solidcore::nurbs::b_spline_curve::BSplineCurve;
    use solidcore::surface::plane::Plane;
    use solidcore::surface::surface::Surface;
    use solidcore::topology::brep_cache::arena_identity;

    /// Concrete curves with metadata describing one unrepresented direct owner,
    /// plus an unrelated surface owner. The metadata's edge list is rotated.
    fn input() -> (Brep, Vec<f64>, [EdgeId; 2], usize) {
        let mut brep = Brep::new();
        brep.set_tolerance(0.0625);
        let vertex = brep.add_vertex(Point3::new(0.0, 0.0, 0.0)).unwrap();
        let first = brep.add_edge(vertex, vertex).unwrap();
        let second = brep.add_edge(vertex, vertex).unwrap();
        for edge in [first, second] {
            brep.edge_mut(edge).unwrap().curve = Some(Curve::from(
                BSplineCurve::create_line_segment(
                    3,
                    Point3::new(0.0, 0.0, 0.0),
                    Point3::new(1.0, 0.0, 0.0),
                )
                .unwrap(),
            ));
        }
        let shell = brep.add_shell().unwrap();
        let face = brep.add_face(shell).unwrap();
        let surface = Surface::from(Plane::from_normal(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ));
        let surface_type = surface.type_code() as f64;
        brep.face_mut(face).unwrap().surface = Some(surface);
        let curve_type = brep
            .edge(first)
            .unwrap()
            .curve
            .as_ref()
            .unwrap()
            .type_code() as f64;
        let mut raw = vec![
            1.0,
            0.125,
            -1.0,
            0.0,
            0.0, // version, tolerance, infinite, regions
            1.0,
            f64::from(face.index()), // faces
            2.0,
            f64::from(second.index()),
            f64::from(first.index()), // edges
            1.0,
            f64::from(vertex.index()), // vertices
            2.0,
            curve_type,
            solidcore::topology::edge::TYPE as f64,
            f64::from(first.index()),
        ];
        let second_record = raw.len();
        raw.extend([curve_type, BREP_TYPE as f64, -3.0]);
        raw.extend([1.0, surface_type, BREP_TYPE as f64, -3.0]);
        raw.extend([1.0, -1.0, 0.0]); // face count, primary use, rectangular trim
        (brep, raw, [first, second], second_record)
    }

    #[test]
    fn malformed_curve_owner_is_an_error_before_metadata_installation() {
        let (mut brep, mut raw, [first, second], record) = input();
        raw[record + 1] = solidcore::topology::edge::TYPE as f64;
        raw[record + 2] = brep.edges().len() as f64; // invalid registered owner
        assert!(apply_query_metadata(&mut brep, &raw, &arena_identity).is_err());
        assert_eq!(brep.tolerance(), 0.0625);
        assert_eq!(brep.edge_list(), &[first, second]);
        assert!(brep
            .edge(first)
            .unwrap()
            .curve
            .as_ref()
            .unwrap()
            .owner()
            .is_none());
    }
}
