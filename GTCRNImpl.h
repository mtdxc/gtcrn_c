
#include <string>
#include <vector>
#include <memory>
#include "onnxruntime_cxx_api.h"

#define SAMEPLERATE (16000)
#define BLOCK_LEN (512)
#define FRAME_LEN (256)
#define FFT_OUT_SIZE (257)

class GTCRNImpl {
public:
  GTCRNImpl(const char* ModelPath);

  int Process(short* in, short* out, int len);
  int Process(float* in, float* out, int len);
  int SampleSize() const { return FRAME_LEN; }
private:
  void onnxInfer();
  void resetInout();

  // OnnxRuntime resources
  Ort::Env env;
  std::shared_ptr<Ort::Session> session = nullptr;
  std::vector<const char*> input_node_names = {"mix", "conv_cache", "tra_cache", "inter_cache"};

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

  float m_windows[BLOCK_LEN] = {0};
};
