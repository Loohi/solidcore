// SPDX-License-Identifier: Apache-2.0
// Query metadata of a real B-rep. Kept separate from export_graph's shared layout.
#pragma once

#include <SmBrep.h>
#include <SmCurve.h>
#include <SmEdge.h>
#include <SmFace.h>
#include <SmRegion.h>
#include <SmSurface.h>
#include <SmTArray.h>
#include <SmVertex.h>

#include <vector>

namespace
{
// Version 1: tolerance, infinite region, raw region void flags; native region,
// face, edge and vertex lists; concrete curve/surface types and direct owners;
// each face's native primary faceuse and raw rectangular-trim flag.
// Each list is length-prefixed. Geometry entries are type, owner type, owner id.
// Null geometry is type -1; null owner is type 0/id -1. An unregistered or
// indirect owner keeps its real type with id -3 for an explicit importer gate.
template <class GraphT>
inline void append_query_metadata(SmBrep* brep, const GraphT& graph, std::vector<double>& out)
{
    out.push_back(1.0);
    out.push_back((double)brep->GetTolerance());
    out.push_back((double)graph.find(0, brep->GetInfiniteRegion()));
    out.push_back((double)graph.regions.size());
    for (SmRegion* region : graph.regions)
        out.push_back((double)region->IsVoid());

    auto put_list = [&](int kind, const auto& list) {
        out.push_back((double)list.GetSize());
        for (ULONG i = 0; i < list.GetSize(); ++i)
            out.push_back((double)graph.find(kind, list[i]));
    };
    SmTArray<SmRegion*> regions;
    SmTArray<SmFace*> faces;
    SmTArray<SmEdge*> edges;
    SmTArray<SmVertex*> vertices;
    brep->GetRegions(regions);
    brep->GetFaces(faces);
    brep->GetEdges(edges);
    brep->GetVertices(vertices);
    put_list(0, regions);
    put_list(3, faces);
    put_list(7, edges);
    put_list(9, vertices);

    auto put_geometry = [&](const auto* geometry, SM_TYPE owner_kind, int registry) {
        const SmObject* owner = geometry ? geometry->GetOwner() : nullptr;
        out.push_back(geometry ? (double)geometry->GetType() : -1.0);
        out.push_back(owner ? (double)owner->GetType() : 0.0);
        out.push_back(owner == nullptr ? -1.0 : owner->GetType() == owner_kind
                                                    ? (double)graph.find(registry, owner)
                                                    : -3.0);
    };
    out.push_back((double)graph.edges.size());
    for (SmEdge* edge : graph.edges)
        put_geometry(edge->GetCurve(), SmEdge_TYPE, 7);
    out.push_back((double)graph.faces.size());
    for (SmFace* face : graph.faces)
        put_geometry(face->GetSurface(), SmFace_TYPE, 3);
    out.push_back((double)graph.faces.size());
    for (SmFace* face : graph.faces)
    {
        out.push_back((double)graph.find(2, face->GetUpwardFaceuse()));
        out.push_back((double)face->GetRectangularTrim());
    }
}
} // namespace
