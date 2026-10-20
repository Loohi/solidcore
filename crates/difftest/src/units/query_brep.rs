//! query.brep: actual `SM_API` evaluation entry points, status and failure atomicity.
//! Surface APIs, decimation attributes and B-rep query batches.
#[path = "query_brep/query_metadata.rs"]
mod query_metadata;

use super::surface_surface_cache_p1::{push3, status};
use super::surface_surface_cache_p3::{gen_surf, push_surf, surf_domain, surface_in, SurfIn};
use crate::case::CaseContext;
use crate::runner::default_cases;
use crate::{Case, HarnessError, Output, Rng, UnitDiff};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use solidcore::geom::vector2d::Vec2;
use solidcore::query::brep;
use solidcore_ffi::units::query_brep as ffi;

/// Native surface API comparison for `query.brep`.
pub struct Runner;
#[derive(Clone, Serialize, Deserialize)]
struct Inputs {
    surf: Option<SurfIn>,
    op: u8,
    uv: [f64; 2],
    /// 0 ordinary; 1/2 NaN u/v; 3/4 infinite u/v. JSON cannot encode NaN.
    #[serde(default)]
    special: u8,
    initial: [[f64; 3]; 3],
}
fn uv(i: &Inputs) -> [f64; 2] {
    let mut uv = i.uv;
    match i.special {
        1 => uv[0] = f64::NAN,
        2 => uv[1] = f64::NAN,
        3 => uv[0] = f64::INFINITY,
        4 => uv[1] = f64::NEG_INFINITY,
        _ => {}
    }
    uv
}
fn flat(i: &Inputs) -> Result<Vec<f64>, HarnessError> {
    let mut out = vec![if i.surf.is_some() { 1.0 } else { 0.0 }];
    if let Some(s) = &i.surf {
        push_surf(&mut out, s)?;
    }
    out.extend(uv(i));
    out.extend(i.initial.iter().flatten());
    Ok(out)
}
fn record(out: Vec<f64>) -> Output {
    Output::new().values("status_and_outputs", out)
}
impl UnitDiff for Runner {
    fn id(&self) -> &'static str {
        "query.brep"
    }
    fn run_cpp(&self, case: &Case) -> Result<Output, HarnessError> {
        if case.inputs.get("bbox_brep").is_some() {
            return bbox_cpp(&case.parse()?);
        }
        if case.inputs.get("point_classify").is_some() {
            let i: PointInput = case.parse()?;
            return ffi::classify(&i.point_classify, i.tolerance, i.boundary, i.regions)
                .map(record)
                .map_err(|e| HarnessError::new(e.to_string()));
        }
        if case.inputs.get("attribute_kind").is_some() {
            return attribute_cpp(&case.parse()?);
        }
        let i: Inputs = case.parse()?;
        let out = ffi::surface(i.op, &flat(&i)?).map_err(|e| HarnessError::new(e.to_string()))?;
        Ok(record(out))
    }
    fn run_rust(&self, case: &Case) -> Result<Output, HarnessError> {
        if case.inputs.get("bbox_brep").is_some() {
            return bbox_rust(&case.parse()?);
        }
        if case.inputs.get("point_classify").is_some() {
            return point_rust(&case.parse()?);
        }
        if case.inputs.get("attribute_kind").is_some() {
            return attribute_rust(&case.parse()?);
        }
        let i: Inputs = case.parse()?;
        let s = i.surf.as_ref().map(surface_in).transpose()?;
        let [u, v] = uv(&i);
        let uv = Vec2::new(u, v);
        let mut result_status = 0.0;
        let mut outputs: Vec<f64> = i.initial.iter().flatten().copied().collect();
        let evaluated = match i.op {
            0 => brep::evaluate_surface_point(s.as_ref(), uv).map(|p| {
                let mut out = Vec::new();
                push3(&mut out, p);
                out
            }),
            1 => brep::evaluate_surface_normal(s.as_ref(), uv).map(|p| {
                let mut out = Vec::new();
                push3(&mut out, p);
                out
            }),
            2 => brep::evaluate_surface_derivatives(s.as_ref(), uv).map(|(p, du, dv)| {
                let mut out = Vec::new();
                push3(&mut out, p);
                push3(&mut out, du);
                push3(&mut out, dv);
                out
            }),
            _ => return Err(HarnessError::new("surface operation")),
        };
        match evaluated {
            Ok(v) => {
                for (target, value) in outputs.iter_mut().zip(v) {
                    *target = value;
                }
            }
            Err(e) => {
                if let Some(deferred) = Output::unported(&e) {
                    return Ok(deferred);
                }
                result_status = status(&e);
            }
        }
        let mut result = vec![result_status];
        result.extend(outputs);
        Ok(record(result))
    }
    fn generate(&self, rng: &mut Rng) -> Option<Value> {
        for _ in 0..100 {
            let surf = gen_surf(rng);
            let [a, b, c, d] = surf_domain(&surf).ok()?;
            let fraction = match rng.below(5) {
                0 => 0.0,
                1 => 1.0,
                _ => rng.unit(),
            };
            let mut uv = [a + fraction * (c - a), b + rng.unit() * (d - b)];
            if rng.below(8) == 0 {
                uv[0] = a - (c - a);
            }
            let i = Inputs {
                surf: Some(surf),
                op: rng.below(3) as u8,
                uv,
                special: if rng.below(10) == 0 {
                    1 + rng.below(4) as u8
                } else {
                    0
                },
                initial: [
                    [17.0, -29.0, 41.0],
                    [53.0, -67.0, 79.0],
                    [83.0, -97.0, 101.0],
                ],
            };
            if ffi::validate(i.op, &flat(&i).ok()?).is_ok() {
                return serde_json::to_value(i).ok();
            }
        }
        None
    }
    fn cases(&self, ctx: &CaseContext) -> Result<Vec<Case>, HarnessError> {
        let mut cases = default_cases(
            self,
            &CaseContext {
                seeds: ctx.seeds.max(600),
                ..ctx.clone()
            },
        )?;
        // Preserve all 600 existing API seeds and add a separate reproducible
        // B-rep box stream; these cases never replace a previous regression.
        for index in 0..ctx.seeds.max(96) {
            let seed = crate::rng::fnv1a64("query.brep.bounding_boxes").wrapping_add(index as u64);
            let input = generate_bbox(&mut Rng::new(seed));
            cases.push(Case {
                unit: self.id().into(),
                name: format!("bbox_gen_{index:04}"),
                source: crate::case::CaseSource::Generated { seed },
                inputs: serde_json::to_value(input)
                    .map_err(|e| HarnessError::new(e.to_string()))?,
            });
        }
        Ok(cases)
    }
}

#[derive(Serialize, Deserialize)]
struct AttributeInput {
    attribute_kind: u8,
    #[serde(default)]
    special: u8,
    id: u64,
    behavior: u8,
    mutate: bool,
    sort: u64,
    class: u8,
    data: [f64; 13],
}
fn attr_record(values: Vec<f64>, integers: [u64; 8]) -> Output {
    // Preserve all 64 identity bits without routing them through a double or
    // overflowing difftest's signed integer representation.
    let integers: Vec<i64> = integers
        .into_iter()
        .flat_map(|v| [i64::from(v as u32), i64::from((v >> 32) as u32)])
        .collect();
    Output::new()
        .values("payloads", values)
        .ints("integer_words", integers)
}
fn attribute_data(i: &AttributeInput) -> [f64; 13] {
    let mut d = i.data;
    match i.special {
        1 => d[0] = f64::NAN,
        2 => d[0] = f64::INFINITY,
        3 => d[0] = f64::NEG_INFINITY,
        _ => {}
    }
    d
}
fn attribute_cpp(i: &AttributeInput) -> Result<Output, HarnessError> {
    let record = solidcore_ffi::units::query_brep_attributes::eval(
        i.attribute_kind,
        i.id,
        i.behavior,
        i.mutate,
        i.sort,
        i.class,
        &attribute_data(i),
    )
    .map_err(|e| HarnessError::new(e.to_string()))?;
    Ok(attr_record(record.values, record.integers))
}
fn attr_payload(a: &solidcore::geom::attribute::Attribute) -> Vec<f64> {
    use solidcore::geom::attribute::AttributeValue;
    let mut out = Vec::new();
    match a.value() {
        AttributeValue::DecimateVertex(v) => {
            out.extend([v.vertex_distance, v.max_face_error]);
            push3(&mut out, v.average_plane_normal);
        }
        AttributeValue::QuadricVertex(v) => {
            for row in v.matrix {
                push3(&mut out, row);
            }
            push3(&mut out, v.linear);
            out.push(v.constant);
        }
        AttributeValue::QuadricEdge(v) => {
            push3(&mut out, v.new_point);
            out.push(v.error);
        }
        _ => {}
    }
    out
}
fn attr_order(a: &solidcore::geom::attribute::Attribute) -> (u64, u64) {
    use solidcore::geom::attribute::AttributeValue;
    match a.value() {
        AttributeValue::DecimateVertex(v) => (v.sort_index, v.vertex_class as u64),
        AttributeValue::QuadricEdge(v) => (v.sort_index.unwrap_or(0), 0),
        _ => (0, 0),
    }
}
fn attribute_rust(i: &AttributeInput) -> Result<Output, HarnessError> {
    use solidcore::geom::attribute::{AttributeValue, Behavior};
    use solidcore::geom::vector3d::Vec3;
    use solidcore::query::brep::PolyVertexClass;
    let behavior =
        Behavior::from_code(i32::from(i.behavior)).map_err(|e| HarnessError::new(e.to_string()))?;
    let mut a = match i.attribute_kind {
        0 => brep::decimate_vertex_attribute(i.id, behavior),
        1 => brep::quadric_vertex_attribute(i.id, behavior),
        2 => brep::quadric_edge_attribute(i.id, behavior),
        _ => return Err(HarnessError::new("attribute kind")),
    };
    let mut values = attr_payload(&a);
    let d = attribute_data(i);
    if i.mutate {
        match a.value_mut() {
            AttributeValue::DecimateVertex(v) => {
                v.sort_index = i.sort;
                v.vertex_class = match i.class {
                    0 => PolyVertexClass::NotComputed,
                    1 => PolyVertexClass::Unknown,
                    2 => PolyVertexClass::Simple,
                    3 => PolyVertexClass::Boundary,
                    4 => PolyVertexClass::Interior,
                    5 => PolyVertexClass::Corner,
                    _ => return Err(HarnessError::new("vertex class")),
                };
                v.vertex_distance = d[0];
                v.max_face_error = d[1];
                v.average_plane_normal = Vec3::new(d[2], d[3], d[4]);
            }
            AttributeValue::QuadricVertex(v) => {
                v.matrix = [
                    Vec3::new(d[0], d[1], d[2]),
                    Vec3::new(d[3], d[4], d[5]),
                    Vec3::new(d[6], d[7], d[8]),
                ];
                v.linear = Vec3::new(d[9], d[10], d[11]);
                v.constant = d[12];
            }
            AttributeValue::QuadricEdge(v) => {
                v.new_point = Vec3::new(d[0], d[1], d[2]);
                v.error = d[3];
                v.sort_index = Some(i.sort);
            }
            _ => return Err(HarnessError::new("attribute value")),
        }
    }
    let copy = a.make_copy();
    values.extend(attr_payload(&a));
    values.extend(attr_payload(&copy));
    let (sort, cls) = attr_order(&a);
    let (csort, ccls) = attr_order(&copy);
    Ok(attr_record(
        values,
        [
            a.attribute_id(),
            copy.attribute_id(),
            a.behavior().code() as u64,
            copy.behavior().code() as u64,
            sort,
            csort,
            cls,
            ccls,
        ],
    ))
}

fn one_region() -> u8 {
    1
}
#[derive(Serialize, Deserialize)]
struct PointInput {
    #[serde(default = "one_region")]
    regions: u8,
    point_classify: [f64; 3],
    tolerance: f64,
    boundary: bool,
}
fn point_rust(i: &PointInput) -> Result<Output, HarnessError> {
    use solidcore::geom::attribute::ObjectKey;
    use solidcore::geom::object::ObjectRef;
    use solidcore::geom::vector3d::Vec3;
    use solidcore::intersect::point_class::PointClassification;
    use solidcore::topology::brep_cache::{arena_identity, BrepCache, CacheObject};
    use solidcore::topology::{Arena, Brep, BrepId, Region};
    use solidcore::EntityRef;
    if !(1..=4).contains(&i.regions) {
        return Err(HarnessError::new(
            "point fixture needs one to four native regions",
        ));
    }
    let k = solidcore::kernel::Kernel::default();
    // Match SmBrep's constructor and the shim's PostInsert calls: one infinite,
    // void region first, then nonvoid empty regions appended in native list order.
    let mut breps: Arena<BrepId, Brep> = Arena::default();
    let brep_id = breps
        .insert(Brep::new())
        .map_err(|e| HarnessError::new(e.to_string()))?;
    let graph = breps
        .get_mut(brep_id)
        .ok_or_else(|| HarnessError::new("registered point-fixture B-rep is missing"))?;
    let brep_ref = ObjectRef {
        type_code: solidcore::geom::a_object::BREP_TYPE,
        key: ObjectKey(u64::from(brep_id.index())),
    };
    graph.set_key(brep_ref.key);
    let infinite = graph
        .add_region_record(Region {
            in_brep: true,
            is_void: 1,
            ..Region::default()
        })
        .map_err(|e| HarnessError::new(e.to_string()))?;
    graph.set_infinite_region_record(Some(infinite));
    graph.set_region_list_head(Some(infinite));
    for _ in 1..i.regions {
        graph
            .add_region_record(Region {
                in_brep: true,
                ..Region::default()
            })
            .map_err(|e| HarnessError::new(e.to_string()))?;
    }
    let keys = |object| match object {
        CacheObject::Brep => brep_ref,
        other => arena_identity(other),
    };
    // This fixture has no curves, surfaces or other geometric owner chains.
    let owner_of = |_: ObjectRef| None;
    let mut cache: Option<BrepCache> = None;
    let mut scene = brep::BrepPointObjects::new(graph, brep_id, &mut cache, &keys, &owner_of);
    let mut pc = PointClassification::new(17.0);
    pc.set_gap_3d(31.0);
    pc.set_t_param(47.0, false);
    pc.set_uv_param(Vec2::new(53.0, 59.0));
    pc.set_pre_snap_param(61.0);
    pc.set_sss_point(Vec3::new(67.0, 71.0, 73.0));
    let tolerance = solidcore::geom::tol::zone_tol3d(&k, i.tolerance);
    let [x, y, z] = i.point_classify;
    let outcome = brep::point_3d_classify(
        &k,
        &mut scene,
        Vec3::new(x, y, z),
        tolerance,
        i.boundary,
        &mut pc,
    );
    let status = match outcome {
        Ok(()) => 0.0,
        Err(e) => {
            if let Some(deferred) = Output::unported(&e) {
                return Ok(deferred);
            }
            status(&e)
        }
    };
    // The native shim compares GetDeviation(), the stored old-tolerance gap.
    let gap = pc.deviation();
    let uv = pc.uv_param();
    let sss = pc.sss_point();
    Ok(record(vec![
        status,
        f64::from(pc.point_class().code()),
        if pc.object() == Some(EntityRef::Region(infinite)) {
            0.0
        } else {
            -1.0
        },
        gap,
        pc.t_param(),
        uv.x,
        uv.y,
        f64::from(u8::from(pc.are_parameters_set())),
        pc.pre_snap_param(),
        f64::from(u8::from(pc.sss_point_flag())),
        sss.x,
        sss.y,
        sss.z,
        pc.src_zone_tol_3d(),
    ]))
}

#[derive(Clone, Serialize, Deserialize)]
struct BboxInput {
    bbox_brep: Option<super::topology_edgeuse_p1::BrepIn>,
    #[serde(default)]
    extra_vertices: usize,
    #[serde(default)]
    extra_edges: usize,
    tolerances: [f64; 3],
    #[serde(default)]
    drop_wire_curves: bool,
    /// Change only the fixture's registry numbering, preserving native lists.
    #[serde(default)]
    reverse_registry: bool,
    steps: Vec<BoxStep>,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum BoxStep {
    Bounding {
        approximate: bool,
        tight: bool,
        expand: bool,
    },
    Tight,
    Lock {
        tight: bool,
    },
    Unlock,
    Default,
    SetEdgeInterval {
        edge_list_index: usize,
        min: f64,
        max: f64,
    },
}
fn bbox_input(i: &BboxInput) -> Result<(String, Vec<f64>), HarnessError> {
    use super::topology_edgeuse_p1::{brep_block_of, path_of};
    let path = path_of(i.bbox_brep.as_ref())?;
    let mut flat = vec![f64::from(u8::from(i.bbox_brep.is_some()))];
    if i.bbox_brep.is_some() {
        flat.extend(brep_block_of(i.bbox_brep.as_ref(), false, false)?);
    }
    flat.extend([i.extra_vertices as f64, i.extra_edges as f64]);
    flat.extend(i.tolerances);
    flat.push(f64::from(u8::from(i.drop_wire_curves)));
    flat.push(f64::from(u8::from(i.reverse_registry)));
    flat.push(i.steps.len() as f64);
    for step in &i.steps {
        let row = match *step {
            BoxStep::Bounding {
                approximate,
                tight,
                expand,
            } => [
                0.0,
                f64::from(u8::from(approximate)),
                f64::from(u8::from(tight)),
                f64::from(u8::from(expand)),
            ],
            BoxStep::Tight => [1.0, 0.0, 0.0, 0.0],
            BoxStep::Lock { tight } => [2.0, 0.0, f64::from(u8::from(tight)), 0.0],
            BoxStep::Unlock => [3.0, 0.0, 0.0, 0.0],
            BoxStep::Default => [4.0, 0.0, 0.0, 0.0],
            BoxStep::SetEdgeInterval {
                edge_list_index,
                min,
                max,
            } => [5.0, edge_list_index as f64, min, max],
        };
        flat.extend(row);
    }
    Ok((path, flat))
}
fn bbox_cpp(i: &BboxInput) -> Result<Output, HarnessError> {
    let (path, flat) = bbox_input(i)?;
    let record = solidcore_ffi::units::query_brep_bbox::eval(false, &path, &flat)
        .map_err(|e| HarnessError::new(e.to_string()))?;
    Ok(Output::new().values("counts_and_boxes", record))
}

fn bbox_index(value: f64, max: usize) -> Result<usize, HarnessError> {
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 || value > max as f64 {
        return Err(HarnessError::new("invalid bbox export count or index"));
    }
    Ok(value as usize)
}

fn bbox_list(
    values: &mut std::slice::Iter<'_, f64>,
    arena_len: usize,
) -> Result<Vec<u32>, HarnessError> {
    let missing = || HarnessError::new("truncated bbox list export");
    let count = bbox_index(*values.next().ok_or_else(missing)?, arena_len)?;
    (0..count)
        .map(|_| {
            let index = bbox_index(
                *values.next().ok_or_else(missing)?,
                arena_len.saturating_sub(1),
            )?;
            u32::try_from(index).map_err(|_| HarnessError::new("bbox list index overflow"))
        })
        .collect()
}

fn bbox_import(
    export: &[f64],
) -> Result<
    (
        solidcore::topology::Brep,
        query_metadata::QueryMetadata,
    ),
    HarnessError,
> {
    use solidcore::topology::{EdgeId, FaceId, VertexId};
    let (length, rest) = export
        .split_first()
        .ok_or_else(|| HarnessError::new("missing bbox export envelope"))?;
    let length = bbox_index(*length, rest.len())?;
    let (graph_export, list_export) = rest.split_at(length);
    let (mut graph, _) = super::topology_edgeuse_p1::import_with_types(graph_export)?;
    let mut values = list_export.iter();
    let faces: Vec<_> = bbox_list(&mut values, graph.faces().len())?
        .into_iter()
        .map(FaceId::from_index)
        .collect();
    let edges: Vec<_> = bbox_list(&mut values, graph.edges().len())?
        .into_iter()
        .map(EdgeId::from_index)
        .collect();
    let vertices: Vec<_> = bbox_list(&mut values, graph.vertices().len())?
        .into_iter()
        .map(VertexId::from_index)
        .collect();
    let metadata_length = bbox_index(
        *values
            .next()
            .ok_or_else(|| HarnessError::new("missing bbox query metadata"))?,
        values.len(),
    )?;
    if metadata_length != values.len() {
        return Err(HarnessError::new(
            "truncated or trailing bbox query metadata",
        ));
    }
    let metadata = query_metadata::apply_query_metadata(
        &mut graph,
        values.as_slice(),
        &query_graph_identity,
    )?;
    if metadata.deferred_output_for(false, false).is_none() {
        use solidcore::topology::brep::p1;
        // These lists are also part of the compared script output. Verify that
        // both export blocks describe the same native ordering before use.
        if p1::faces(&graph, None).map_err(|e| HarnessError::new(e.to_string()))? != faces
            || p1::edges(&graph, None).map_err(|e| HarnessError::new(e.to_string()))? != edges
            || p1::vertices(&graph, None).map_err(|e| HarnessError::new(e.to_string()))? != vertices
        {
            return Err(HarnessError::new("bbox lists disagree with query metadata"));
        }
    }
    Ok((graph, metadata))
}

fn bbox_rust(i: &BboxInput) -> Result<Output, HarnessError> {
    use solidcore::geom::extent3d::Extent3d;
    use solidcore::geom::vector3d::Vec3;
    use solidcore::topology::brep::p1;
    let (path, flat) = bbox_input(i)?;
    let export = solidcore_ffi::units::query_brep_bbox::eval(true, &path, &flat)
        .map_err(|e| HarnessError::new(e.to_string()))?;
    let (mut graph, metadata) = bbox_import(&export)?;
    if let Some(deferred) = metadata.deferred_output_for(false, false) {
        return Ok(deferred);
    }
    let k = solidcore::kernel::Kernel::default();
    let vertices = p1::vertices(&graph, None).map_err(|e| HarnessError::new(e.to_string()))?;
    let edges = p1::edges(&graph, None).map_err(|e| HarnessError::new(e.to_string()))?;
    let faces = p1::faces(&graph, None).map_err(|e| HarnessError::new(e.to_string()))?;
    let mut record = vec![
        vertices.len() as f64,
        edges.len() as f64,
        faces.len() as f64,
    ];
    record.push(faces.len() as f64);
    record.extend(faces.iter().map(|id| f64::from(id.index())));
    record.push(edges.len() as f64);
    record.extend(edges.iter().map(|id| f64::from(id.index())));
    record.push(vertices.len() as f64);
    record.extend(vertices.iter().map(|id| f64::from(id.index())));
    let initial =
        Extent3d::from_points(Vec3::new(-17.0, -19.0, -23.0), Vec3::new(29.0, 31.0, 37.0));
    let mut locked = false;
    for step in &i.steps {
        // Guard only geometry actually read by this step. Approximation may
        // skip faces and, above both native thresholds, edges too. A stored lock
        // bypasses ordinary BoundingBox, while Tight and Lock recompute.
        let (read_curves, read_surfaces) = match *step {
            BoxStep::Bounding { approximate, .. } if !locked => (
                !(approximate && vertices.len() > 500 && edges.len() > 100),
                !approximate,
            ),
            BoxStep::Default if !locked => (true, true),
            BoxStep::Tight | BoxStep::Lock { .. } => (true, true),
            BoxStep::SetEdgeInterval { .. } => (true, false),
            _ => (false, false),
        };
        if let Some(deferred) = metadata.deferred_output_for(read_curves, read_surfaces) {
            return Ok(deferred);
        }
        let result = match *step {
            BoxStep::Bounding {
                approximate,
                tight,
                expand,
            } => brep::calculate_bounding_box(
                &k,
                &graph,
                brep::BoundingBoxOptions {
                    approximate,
                    tight,
                    expand_by_zone_tolerance: expand,
                },
            ),
            BoxStep::Tight => brep::calculate_tight_bounding_box(&k, &graph),
            BoxStep::Lock { tight } => {
                locked = false;
                brep::lock_bounding_box(&k, &mut graph, tight).map(|()| {
                    locked = true;
                    initial
                })
            }
            BoxStep::Unlock => {
                brep::unlock_bounding_box(&mut graph);
                locked = false;
                Ok(initial)
            }
            BoxStep::Default => brep::calculate_loose_bounding_box(&k, &graph, false, true),
            BoxStep::SetEdgeInterval {
                edge_list_index,
                min,
                max,
            } => {
                let edge = edges
                    .get(edge_list_index)
                    .copied()
                    .ok_or_else(|| HarnessError::new("bbox script edge outside native list"))?;
                solidcore::topology::edge::set_interval(
                    &mut graph,
                    edge,
                    &solidcore::geom::extent1d::Extent1d::from_min_max(min, max),
                )
                .map(|()| initial)
            }
        };
        let bbox = match result {
            Ok(bbox) => {
                record.push(0.0);
                bbox
            }
            Err(e) => {
                if let Some(deferred) = Output::unported(&e) {
                    return Ok(deferred);
                }
                record.push(status(&e));
                initial
            }
        };
        push3(&mut record, bbox.min());
        push3(&mut record, bbox.max());
    }
    Ok(Output::new().values("counts_and_boxes", record))
}
fn generate_bbox(rng: &mut Rng) -> BboxInput {
    use super::topology_edgeuse_p1::{gen_curve, BrepIn};
    let scales = [1e-6, 1e-3, 1.0, 1e3, 1e6];
    let scale = *rng.choose(&scales).unwrap_or(&1.0);
    let curve = gen_curve(rng, 3, scale);
    let a = curve.u.first().copied().unwrap_or(0.0);
    let b = curve.u.last().copied().unwrap_or(1.0);
    let t0 = a + (b - a) * rng.uniform(0.0, 0.2);
    let t1 = b - (b - a) * rng.uniform(0.0, 0.2);
    let tight = rng.chance(0.25);
    BboxInput {
        bbox_brep: Some(BrepIn::Wire { curve, t0, t1 }),
        extra_vertices: rng.below(4) as usize,
        extra_edges: rng.below(3) as usize,
        tolerances: [rng.unit() * 1e-4, rng.unit() * 1e-3, 0.0],
        drop_wire_curves: rng.chance(0.1),
        reverse_registry: false,
        steps: vec![
            BoxStep::Default,
            BoxStep::Bounding {
                approximate: rng.chance(0.5),
                tight,
                expand: rng.chance(0.5),
            },
            BoxStep::Lock { tight },
            BoxStep::Bounding {
                approximate: true,
                tight: !tight,
                expand: false,
            },
            BoxStep::Unlock,
            BoxStep::Tight,
        ],
    }
}

/// The identity the bbox fixtures' native metadata is keyed by: the arena identity
/// shifted past the importer's own keys (as the shared query-graph importers do).
fn query_graph_identity(
    object: solidcore::topology::brep_cache::CacheObject,
) -> solidcore::geom::object::ObjectRef {
    let mut identity = solidcore::topology::brep_cache::arena_identity(object);
    identity.key = solidcore::geom::attribute::ObjectKey(identity.key.0 + 0x10_0000);
    identity
}
