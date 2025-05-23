
#include <string>
#include <vector>
#include <memory>
#include "onnxruntime_cxx_api.h"

#define SAMEPLERATE (16000)
#define BLOCK_LEN (512)
#define BLOCK_SHIFT (256)
#define FFT_OUT_SIZE (257)

class GTCRNImpl {
public:
  GTCRNImpl(const char* ModelPath);

  int Enhance(float* in, float* out, int len);

  void OnnxInfer();
private:
  // OnnxRuntime resources
  Ort::Env env;
  std::shared_ptr<Ort::Session> session = nullptr;
  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU);

  std::vector<Ort::Value> ort_inputs;
  std::vector<const char*> input_node_names = {"mix", "conv_cache", "tra_cache", "inter_cache"};

  std::vector<Ort::Value> ort_outputs;
  std::vector<const char*> output_node_names = {"enh", "conv_cache_out", "tra_cache_out", "inter_cache_out"};

  const int64_t infea_node_dims[4] = {1, FFT_OUT_SIZE, 1, 2};
  const int64_t conv_cache_dims[5] = {2, 1, 16, 16, 33};
  const int64_t tra_cache_dims[5] = {2, 3, 1, 1, 16};
  const int64_t inter_cache_dims[4] = {2, 1, 33, 16};
  float mic_buffer_[BLOCK_LEN] = { 0 };
  float out_buffer_[BLOCK_LEN] = { 0 };
  float conv_cache_[2 * 16 * 16 * 33] = { 0 };
  float tra_cache_[2 * 3 * 16] = { 0 };
  float inter_cache_[2 * 33 * 16] = { 0 };
  void ResetInout();

  float m_windows[BLOCK_LEN] = {0};
};
