// SPDX-License-Identifier: Apache-2.0
// Real B-rep bounding-box operations. Construction/export reuses the common
// traversal reader; export never calls a bounding-box operation under test.
#include "sc_edgeuse_graph.h"
#include "sc_brep_query_metadata.h"
#include <array>
namespace {
// Complete the registry independently of the region/shell traversal. Its indices
// identify objects; they are not the B-rep's native list order. Selected fixtures
// deliberately register the lists backwards to make that distinction observable.
void bbox_walk(SmBrep* brep,Graph& graph,bool reverse_registry) {
    SmTArray<SmVertex*> vertices;brep->GetVertices(vertices);
    SmTArray<SmEdge*> edges;brep->GetEdges(edges);
    SmTArray<SmFace*> faces;brep->GetFaces(faces);
    if(reverse_registry) {
        for(ULONG i=vertices.GetSize();i>0;--i) graph.vertex(vertices[i-1]);
        for(ULONG i=edges.GetSize();i>0;--i) graph.edge(edges[i-1]);
        for(ULONG i=faces.GetSize();i>0;--i) graph.face(faces[i-1]);
    }
    walk(brep,graph);
    for(ULONG i=0;i<vertices.GetSize();i++) graph.vertex(vertices[i]);
    for(ULONG i=0;i<edges.GetSize();i++) graph.edge(edges[i]);
    for(ULONG i=0;i<faces.GetSize();i++) graph.face(faces[i]);
    // Resolve use records and links of newly registered owned objects too.
    walk(brep,graph);
}

// Complete native lists as registry indices, each preceded by its count. No
// sorting, reachable-only filtering, or invented membership is permitted here.
bool put_bbox_lists(SmBrep* brep,Graph& graph,std::vector<double>& out) {
    SmTArray<SmFace*> faces;brep->GetFaces(faces);
    SmTArray<SmEdge*> edges;brep->GetEdges(edges);
    SmTArray<SmVertex*> vertices;brep->GetVertices(vertices);
    out.push_back((double)faces.GetSize());
    for(ULONG i=0;i<faces.GetSize();++i) {
        int64_t id=graph.find(3,faces[i]);if(id<0) return false;out.push_back((double)id);
    }
    out.push_back((double)edges.GetSize());
    for(ULONG i=0;i<edges.GetSize();++i) {
        int64_t id=graph.find(7,edges[i]);if(id<0) return false;out.push_back((double)id);
    }
    out.push_back((double)vertices.GetSize());
    for(ULONG i=0;i<vertices.GetSize();++i) {
        int64_t id=graph.find(9,vertices[i]);if(id<0) return false;out.push_back((double)id);
    }
    return true;
}
}
SC_EXPORT int32_t sc_query_brep_bbox(int32_t op,const char* path,const double* input,int64_t n_in,
    double* out,int64_t cap,int64_t* out_len) {
    if(op<0 || op>1 || !input || n_in<9) return SC_ERR_ARG;
    SC_TRY
    Rd rd{input,0}; Built built; SmStatus status=SM_SUCCESS;
    const SmContext& ctx=the_context();
    if(rd.flag()) built.brep=build(rd,path,status);
    else built.brep=new (ctx) SmBrep();
    if(!built.brep || status!=SM_SUCCESS) return status==SM_SUCCESS?SC_ERR_ARG:(int32_t)status;
    SmBrep* brep=built.brep;
    int64_t nv=rd.idx(),ne=rd.idx();
    double vt=rd.num(),et=rd.num(),ft=rd.num();
    bool drop=rd.flag();
    bool reverse_registry=rd.flag();
    for(int64_t i=0;i<ne;i++) {
        SmPoint3d a((double)i*0.01,0.1,0.0),b((double)i*0.01+1.0,0.1,0.0);
        SmBSplineCurve* curve=nullptr;SmEdge* edge=nullptr;
        status=SmBSplineCurve::CreateLineSegment(ctx,3,a,b,curve);
        if(status!=SM_SUCCESS) return (int32_t)status;
        status=brep->CreateWireEdgeFromCurve(curve,curve->GetNaturalInterval(),edge);
        if(status!=SM_SUCCESS) return (int32_t)status;
    }
    for(int64_t i=0;i<nv;i++) {
        SmShell* shell=nullptr;SmVertex* vertex=nullptr;
        status=brep->MakeShellVertex(brep->GetInfiniteRegion(),SmPoint3d((double)i*0.01,(double)(i%7)*0.001,0.0),shell,vertex);
        if(status!=SM_SUCCESS) return (int32_t)status;
    }
    Graph graph;bbox_walk(brep,graph,reverse_registry);
    for(auto* v:graph.vertices) v->SetTolerance(SmZoneTol3d(vt),FALSE,FALSE);
    for(auto* e:graph.edges) e->SetTolerance(SmZoneTol3d(et),FALSE,FALSE);
    for(auto* f:graph.faces) f->SetTolerance(SmZoneTol3d(ft),FALSE,FALSE);
    if(drop) for(auto* e:graph.edges) {
        SmTArray<SmFace*> faces;e->GetFaces(faces);
        if(faces.GetSize()==0) e->SetCurve(NULL,TRUE,FALSE,FALSE);
    }
    int64_t count=rd.idx();std::vector<std::array<double,4>> steps;
    for(int64_t i=0;i<count;i++) {
        double code=rd.num(),a=rd.num(),b=rd.num(),c=rd.num();
        steps.push_back({code,a,b,c});
    }
    if(rd.at!=n_in) return SC_ERR_ARG;
    std::vector<double> result;
    if(op==0) {
        // Envelope: graph length, unchanged common graph export, the three
        // native lists, then length-prefixed query metadata. The metadata keeps
        // exact primary faceuses, trim flags, owners and concrete types.
        Graph final_graph;bbox_walk(brep,final_graph,reverse_registry);
        std::vector<double> exported{0.0};
        export_graph(brep,final_graph,exported);
        result.push_back((double)exported.size());
        result.insert(result.end(),exported.begin(),exported.end());
        if(!put_bbox_lists(brep,final_graph,result)) return SC_ERR_ARG;
        std::vector<double> metadata;
        append_query_metadata(brep,final_graph,metadata);
        result.push_back((double)metadata.size());
        result.insert(result.end(),metadata.begin(),metadata.end());
    } else {
        SmTArray<SmVertex*> vertices;brep->GetVertices(vertices);
        SmTArray<SmEdge*> edges;brep->GetEdges(edges);
        SmTArray<SmFace*> faces;brep->GetFaces(faces);
        result.push_back((double)vertices.GetSize());result.push_back((double)edges.GetSize());
        result.push_back((double)faces.GetSize());
        if(!put_bbox_lists(brep,graph,result)) return SC_ERR_ARG;
        for(const auto& step:steps) {
            SmExtent3d box;box.SetMinMax(SmPoint3d(-17,-19,-23),SmPoint3d(29,31,37));
            if(step[0]==0) status=brep->CalculateBoundingBox(box,(SmBoolean)step[1],(SmBoolean)step[2],(SmBoolean)step[3]);
            else if(step[0]==1) status=brep->CalculateTightBoundingBox(box);
            else if(step[0]==2) status=brep->LockBoundingBox((SmBoolean)step[2]);
            else if(step[0]==3) {brep->UnLockBoundingBox();status=SM_SUCCESS;}
            else if(step[0]==4) status=brep->CalculateBoundingBox(box);
            else if(step[0]==5) {
                // Real SetInterval; identify the target by native list position,
                // independently of the export registry's arena numbering.
                ULONG index=(ULONG)step[1];
                if(index>=edges.GetSize()) return SC_ERR_ARG;
                edges[index]->SetInterval(SmExtent1d(step[2],step[3]));
                status=SM_SUCCESS;
            }
            else return SC_ERR_ARG;
            result.push_back(st(status));put_vec(result,box.GetMin());put_vec(result,box.GetMax());
        }
    }
    return sc_copy_out(result.data(),(int64_t)result.size(),out,cap,out_len);
    SC_CATCH
}
