// Minimal C ABI: causal GGUF backbone + original DecisionLFM head, or native CLEF head.
#include "llama.h"
#include "llama-ext.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

struct Engine {
    llama_model *model = nullptr;
    llama_context *ctx = nullptr;
    ggml_context *head = nullptr;
    bool clef = false;
    int dim = 0;
    ~Engine() { if(ctx) llama_free(ctx); if(model) llama_model_free(model); if(head) ggml_free(head); }
};
static thread_local std::string last_error;
static void quiet(enum ggml_log_level level, const char *msg, void *) {
    if(level >= GGML_LOG_LEVEL_WARN) fputs(msg, stderr);
}
extern "C" const char *df_error() { return last_error.c_str(); }
extern "C" void *df_create(const char *path, const char *head, int capacity) {
    Engine *e = new Engine;
    try {
        llama_log_set(quiet, nullptr);
        llama_backend_init();
        auto mp = llama_model_default_params(); mp.n_gpu_layers = -1;
        e->model = llama_model_load_from_file(path, mp);
        if(!e->model) throw std::runtime_error("GGUF model load failed");
        e->clef = !head || !*head;
        e->dim = llama_model_n_embd(e->model);
        auto cp = llama_context_default_params();
        cp.n_ctx = cp.n_batch = cp.n_ubatch = capacity;
        cp.n_seq_max = 1; cp.n_threads = cp.n_threads_batch = 8;
        cp.embeddings = true; cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
        cp.attention_type = LLAMA_ATTENTION_TYPE_CAUSAL;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.no_perf = true;
        e->ctx = llama_init_from_model(e->model, cp);
        if(!e->ctx) throw std::runtime_error("GGUF context creation failed");
        if(!e->clef) {
            auto f = gguf_init_from_file(head, {false, &e->head});
            if(!f || !e->head) throw std::runtime_error("head GGUF load failed");
            gguf_free(f);
            for(const char *name: {"norm.weight", "norm.bias", "query.weight", "option.weight", "log_scale"}) {
                if(!ggml_get_tensor(e->head,name)) throw std::runtime_error(std::string("missing head tensor: ")+name);
            }
            auto norm=ggml_get_tensor(e->head,"norm.weight");
            auto bias=ggml_get_tensor(e->head,"norm.bias");
            auto query=ggml_get_tensor(e->head,"query.weight");
            auto option=ggml_get_tensor(e->head,"option.weight");
            if(norm->type!=GGML_TYPE_F32 || bias->type!=GGML_TYPE_F32 || norm->ne[0]!=e->dim
                    || bias->ne[0]!=e->dim || query->ne[0]!=e->dim || option->ne[0]!=e->dim
                    || query->ne[1]!=option->ne[1]) throw std::runtime_error("head dimensions or norms incompatible with backbone");
        }
        return e;
    } catch(const std::exception &ex) { last_error = ex.what(); delete e; return nullptr; }
}
extern "C" void df_free(void *ptr) { delete static_cast<Engine *>(ptr); }
extern "C" void df_sync(void *ptr) { llama_synchronize(static_cast<Engine *>(ptr)->ctx); }
extern "C" int df_eval(void *ptr, const int32_t *ids, const int32_t *orders,
    int n, const int32_t *option_map, int count, float *scores) {
    auto *e = static_cast<Engine *>(ptr);
    llama_batch_ext *batch = nullptr;
    ggml_context *work = nullptr;
    try {
        if(n <= 0 || count < 2) throw std::runtime_error("invalid input dimensions");
        auto memory = llama_get_memory(e->ctx);
        if(memory) llama_memory_clear(memory, true);
        batch = llama_batch_ext_init(e->ctx);
        for(int i=0; i<n; ++i) {
            int j=llama_batch_ext_add_token(batch, 0, ids[i]);
            llama_pos pos=i;
            if(j<0 || !llama_batch_ext_set_pos(batch,j,&pos)) throw std::runtime_error("batch position failed");
            bool output = true; // Native CLEF pads scores to token rows; LFM returns all hidden states.
            llama_batch_ext_set_output_embd(batch,j,output);
            if(e->clef && !llama_batch_ext_set_decision_order(batch,j,(llama_decision_order)orders[i]))
                throw std::runtime_error("decision span failed");
        }
        int status = llama_process(e->ctx, LLAMA_PROCESS_TYPE_DECODE, batch);
        llama_batch_ext_free(batch); batch=nullptr;
        if(status) throw std::runtime_error("llama_process status="+std::to_string(status));
        llama_synchronize(e->ctx);
        const float *hidden = llama_get_embeddings(e->ctx);
        if(!hidden) throw std::runtime_error("missing embeddings/scores");
        if(e->clef) { std::copy(hidden,hidden+count,scores); return 0; }
        int d=e->dim;
        std::vector<float> pooled((count+1)*d,0);
        std::copy(hidden+(n-1)*d, hidden+n*d, pooled.begin());
        std::vector<int> lengths(count,0);
        for(int t=0;t<n;++t) {
            int k=option_map[t]; if(k<0) continue;
            if(k>=count) throw std::runtime_error("invalid option index");
            ++lengths[k];
            for(int j=0;j<d;++j) pooled[(k+1)*d+j]+=hidden[t*d+j];
        }
        auto weight = ggml_get_tensor(e->head,"norm.weight");
        auto bias = ggml_get_tensor(e->head,"norm.bias");
        const float *w=ggml_get_data_f32(weight), *b=ggml_get_data_f32(bias);
        for(int k=0;k<=count;++k) {
            float *x=pooled.data()+k*d;
            if(k>0) {
                if(!lengths[k-1]) throw std::runtime_error("empty option span");
                for(int j=0;j<d;++j) x[j]/=lengths[k-1];
            }
            double mean=0, var=0;
            for(int j=0;j<d;++j) mean+=x[j]; mean/=d;
            for(int j=0;j<d;++j) var+=(x[j]-mean)*(x[j]-mean); var/=d;
            float inv=1/std::sqrt(var+1e-5);
            for(int j=0;j<d;++j) x[j]=(x[j]-mean)*inv*w[j]+b[j];
        }
        work=ggml_init({size_t(16*1024*1024),nullptr,false});
        auto input=ggml_new_tensor_2d(work,GGML_TYPE_F32,d,count+1);
        std::memcpy(input->data,pooled.data(),pooled.size()*sizeof(float));
        auto qin=ggml_view_2d(work,input,d,1,d*sizeof(float),0);
        auto kin=ggml_view_2d(work,input,d,count,d*sizeof(float),d*sizeof(float));
        auto q=ggml_mul_mat(work,ggml_get_tensor(e->head,"query.weight"),qin);
        auto k=ggml_mul_mat(work,ggml_get_tensor(e->head,"option.weight"),kin);
        auto graph=ggml_new_graph(work);
        ggml_build_forward_expand(graph,q); ggml_build_forward_expand(graph,k);
        if(ggml_graph_compute_with_ctx(work,graph,8)!=GGML_STATUS_SUCCESS) throw std::runtime_error("head compute failed");
        int h=q->ne[0];
        const float *qv=ggml_get_data_f32(q), *kv=ggml_get_data_f32(k);
        double qnorm=0; for(int j=0;j<h;++j) qnorm+=qv[j]*qv[j];
        qnorm=std::max(std::sqrt(qnorm),1e-12);
        float scale=std::min(std::exp(*ggml_get_data_f32(ggml_get_tensor(e->head,"log_scale"))),100.0f);
        for(int i=0;i<count;++i) {
            double dot=0, knorm=0;
            for(int j=0;j<h;++j) { dot+=qv[j]*kv[i*h+j]; knorm+=kv[i*h+j]*kv[i*h+j]; }
            scores[i]=scale*dot/(qnorm*std::max(std::sqrt(knorm),1e-12));
        }
        ggml_free(work); return 0;
    } catch(const std::exception &ex) {
        last_error=ex.what(); if(batch) llama_batch_ext_free(batch); if(work) ggml_free(work); return -1;
    }
}
extern "C" unsigned long long df_vram(void *ptr) {
    auto *e=static_cast<Engine *>(ptr);
    unsigned long long bytes=0;
    for(const auto &item: llama_get_memory_breakdown(e->ctx)) {
        auto dev=ggml_backend_buft_get_device(item.first);
        if(dev && !ggml_backend_buft_is_host(item.first) && ggml_backend_dev_type(dev)==GGML_BACKEND_DEVICE_TYPE_GPU) bytes+=item.second.total();
    }
    return bytes;
}

