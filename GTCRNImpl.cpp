
#include "GTCRNImpl.h"
#include "pocketfft_hdronly.h"

#ifndef PI
#define PI 3.14159265358979323846f
#endif
#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))

GTCRNImpl::GTCRNImpl(const char *ModelPath)
{
  // Init threads = 1
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.SetInterOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  // Load model
  std::vector<uint8_t> model;
  if (FILE* fp = fopen(ModelPath, "rb")) {
      fseek(fp, 0, SEEK_END);
      model.resize(ftell(fp));
      fseek(fp, 0, SEEK_SET);
      fread(model.data(), model.size(), 1, fp);
      fclose(fp);
      session = std::make_shared<Ort::Session>(env, model.data(), model.size(), session_options);
  }
  for (int i = 0; i < BLOCK_LEN; i++) {
    m_windows[i] = sinf(PI * i / (BLOCK_LEN - 1));
  }
  resetInout();
}

void GTCRNImpl::resetInout()
{
  memset(mic_buffer_, 0, sizeof(mic_buffer_));
  memset(out_buffer_, 0, sizeof(out_buffer_));
  memset(conv_cache_, 0, sizeof(conv_cache_));
  memset(tra_cache_, 0, sizeof(tra_cache_));
  memset(inter_cache_, 0, sizeof(inter_cache_));
}

int GTCRNImpl::Process(float* in, float* out, int len) {
    if (len != FRAME_LEN) {
        return -1;
    }
    memmove(mic_buffer_, mic_buffer_ + FRAME_LEN, FRAME_LEN * sizeof(float));
    memcpy(mic_buffer_ + FRAME_LEN, in, sizeof(float) * FRAME_LEN);
    onnxInfer();
    memcpy(out, out_buffer_, sizeof(float) * FRAME_LEN);
    return FRAME_LEN;
}

int GTCRNImpl::Process(short *in, short*out, int len)
{
  if (len != FRAME_LEN) {
    return -1;
  }
  memmove(mic_buffer_, mic_buffer_ + FRAME_LEN, FRAME_LEN * sizeof(float));
  for (int n = 0; n < FRAME_LEN; n++)
    mic_buffer_[n + FRAME_LEN] = in[n] / 32767.0;

  onnxInfer();
  for (int j = 0; j < FRAME_LEN; j++)
    out[j] = out_buffer_[j] * 32768; // for one forward process save first FRAME_LEN model output samples
  return FRAME_LEN;
}

void GTCRNImpl::onnxInfer()
{
  typedef std::complex<float> cpx_type;
  std::vector<cpx_type> mic_res(BLOCK_LEN);

  std::vector<size_t> shape = { BLOCK_LEN };
  std::vector<size_t> axes = {0};
  std::vector<ptrdiff_t> stridel = { sizeof(float) };
  std::vector<ptrdiff_t> strideo = {sizeof(cpx_type)};

  float fft_in[BLOCK_LEN];
  for (int i = 0; i < BLOCK_LEN; i++) {
    fft_in[i] = mic_buffer_[i] * m_windows[i];
  }

  pocketfft::r2c(shape, stridel, strideo, axes, pocketfft::FORWARD, fft_in, mic_res.data(), 1.0f);

  float mic_fea[FFT_OUT_SIZE * 2] = {0};
  for (int i = 0; i < FFT_OUT_SIZE; i++)
  {
    mic_fea[2 * i] = mic_res[i].real();
    mic_fea[2 * i + 1] = mic_res[i].imag();
  }

  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  std::vector<Ort::Value> ort_inputs;
  ort_inputs.emplace_back(Ort::Value::CreateTensor<float>(
      memory_info, mic_fea, FFT_OUT_SIZE * 2, infea_node_dims, 4));
  ort_inputs.emplace_back(Ort::Value::CreateTensor<float>(
      memory_info, conv_cache_, 2 * 16 * 16 * 33, conv_cache_dims, 5));
  ort_inputs.emplace_back(Ort::Value::CreateTensor<float>(
      memory_info, tra_cache_, 2 * 3 * 16, tra_cache_dims, 5));
  ort_inputs.emplace_back(Ort::Value::CreateTensor<float>(
      memory_info, inter_cache_, 2 * 33 * 16, inter_cache_dims, 4));

  std::vector<Ort::Value> ort_outputs = session->Run(Ort::RunOptions{nullptr},
                             input_node_names.data(), ort_inputs.data(), ort_inputs.size(),
                             output_node_names.data(), output_node_names.size());

  float *out_concache = ort_outputs[1].GetTensorMutableData<float>();
  std::memcpy(conv_cache_, out_concache, 2 * 16 * 16 * 33 * sizeof(float));

  float *out_tracache = ort_outputs[2].GetTensorMutableData<float>();
  std::memcpy(tra_cache_, out_tracache, 2 * 3 * 16 * sizeof(float));

  float *out_intercache = ort_outputs[3].GetTensorMutableData<float>();
  std::memcpy(inter_cache_, out_intercache, 2 * 33 * 16 * sizeof(float));

  float *out_fea = ort_outputs[0].GetTensorMutableData<float>();
  for (int i = 0; i < FFT_OUT_SIZE; i++) {
    mic_res[i] = cpx_type(out_fea[2 * i], out_fea[2 * i + 1]);
  }
  float mic_in[BLOCK_LEN];
  pocketfft::c2r(shape, strideo, stridel, axes, pocketfft::BACKWARD, mic_res.data(), mic_in, 1.0f);

  float estimated_block[BLOCK_LEN];
  for (int i = 0; i < BLOCK_LEN; i++)
    estimated_block[i] = mic_in[i] / BLOCK_LEN * m_windows[i];

  memmove(out_buffer_, out_buffer_ + FRAME_LEN, FRAME_LEN * sizeof(float));
  memset(out_buffer_ + FRAME_LEN, 0, FRAME_LEN * sizeof(float));
  for (int i = 0; i < BLOCK_LEN; i++) {
    out_buffer_[i] += estimated_block[i];
  }
}
