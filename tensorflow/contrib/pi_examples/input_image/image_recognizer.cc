//
// Created by kerner on 10/16/17.
//

#include <fstream>
#include <string>
#include <iomanip>

#include "image_recognizer.h"

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace PI {
namespace recognition {
class ImageRecognizer::Impl {
 public:
  // Impl start
  Impl(const Parameters &params);
  ~Impl();

  std::vector<ObjectInfo> Recognize(
      uint8_t *image_data, const int image_width, const int image_height,
      const int image_channels);

 private:
  // graph to be executed
  std::string graph;
  // name of file containing labels
  std::string labels;
  // resize image to this width in pixels
  int32_t input_width;
  // resize image to this height in pixels
  int32_t input_height;
  // scale pixel values to this mean
  int32_t input_mean;
  // scale pixel values to this std deviation
  int32_t input_std;
  // name of input layer
  std::string input_layer;
  // name of output layers
  std::vector<std::string> output_layers;
  // cpu cores
  int32_t cores;

  std::unique_ptr<tensorflow::Session> session;
  std::vector<std::string> label_list;
  size_t label_count;

  class ImageRecognizer;
  tensorflow::Status ReadTensorFromImage(
      uint8_t *image_data, int image_width, int image_height,
      int image_channels, std::vector<tensorflow::Tensor> *out_tensors);
  tensorflow::Status GetTopLabels(
      const std::vector<tensorflow::Tensor> &outputs, int how_many_labels,
      tensorflow::Tensor *out_indices, tensorflow::Tensor *out_scores);
  tensorflow::Status LoadGraph(std::string graph_file_name,
                               std::unique_ptr<tensorflow::Session> *session);
  tensorflow::Status ReadLabelsFile(std::string file_name,
                                    std::vector<std::string> *result,
                                    size_t *found_label_count);
};

ImageRecognizer::Impl::Impl(const Parameters &params)
    : graph(params.graph),
      labels(params.labels),
      input_width(params.input_width),
      input_height(params.input_height),
      input_mean(params.input_mean),
      input_std(params.input_std),
      input_layer(params.input_layer),
      output_layers(params.output_layers),
      cores(params.cores) {
  // First we load and initialize the model.
  std::string graph_path = tensorflow::io::JoinPath("", graph);
  tensorflow::Status load_graph_status = LoadGraph(graph_path, &session);
  if (!load_graph_status.ok()) {
    std::cerr << load_graph_status << std::endl;
  }

  tensorflow::Status read_labels_status =
      ReadLabelsFile(labels, &label_list, &label_count);
  if (!read_labels_status.ok()) {
    std::cerr << read_labels_status << std::endl;
  }
}

ImageRecognizer::Impl::~Impl() {}

std::vector<ObjectInfo> ImageRecognizer::Impl::Recognize(
    uint8_t *image_data, const int image_width, const int image_height,
    const int image_channels) {
  std::vector<ObjectInfo> top_result;
  // Get the image from image_data as a float array of numbers, resized and
  // normalized
  // to the specifications the main graph expects.
  std::vector<tensorflow::Tensor> resized_tensors;
  tensorflow::Status read_tensor_status = ReadTensorFromImage(
      image_data, image_width, image_height, image_channels, &resized_tensors);
  if (!read_tensor_status.ok()) {
    std::cerr << read_tensor_status << std::endl;
    return top_result;
  }
  const tensorflow::Tensor &resized_tensor = resized_tensors[0];

  // Actually run the image through the model.
  std::vector<tensorflow::Tensor> outputs;
  tensorflow::Status run_status =
      session->Run({{input_layer, resized_tensor}}, output_layers, {}, &outputs);
  if (!run_status.ok()) {
    std::cerr << "Running model failed: " << run_status << std::endl;
  }

  // object detect
  if (outputs.size() >= 4) {
    tensorflow::TTypes<float>::Flat scores = outputs[1].flat<float>();
    tensorflow::TTypes<float>::Flat classes = outputs[2].flat<float>();
    tensorflow::TTypes<float>::Flat num_detections = outputs[3].flat<float>();
    auto boxes = outputs[0].flat_outer_dims<float, 3>();

    // LOG(INFO) << "num_detections:" << num_detections(0) << ","
    // << outputs[0].shape().DebugString();

    for (size_t i = 0; i < num_detections(0) && i < 20; ++i) {
      if (scores(i) > 0.5) {
        // LOG(INFO) << i << ",score:" << scores(i) << ",class:" << classes(i)
        // << ",box:" << boxes(0, i, 0) << "," << boxes(0, i, 1) << ","
        // << boxes(0, i, 2) << "," << boxes(0, i, 3);
        top_result.push_back(
            {label_list[(int) (classes(i)) - 1], scores(i),
             boxes(0, i, 1) * image_width, boxes(0, i, 0) * image_height,
             boxes(0, i, 3) * image_width, boxes(0, i, 2) * image_height});
      }
    }
  } else {  // classify
    const int how_many_labels = std::min(5, static_cast<int>(label_count));
    tensorflow::Tensor indices;
    tensorflow::Tensor scores;
    tensorflow::Status top_label_result =
        GetTopLabels(outputs, how_many_labels, &indices, &scores);
    if (!top_label_result.ok()) {
      return top_result;
    }
    tensorflow::TTypes<float>::Flat scores_flat = scores.flat<float>();
    tensorflow::TTypes<tensorflow::int32>::Flat indices_flat =
        indices.flat<tensorflow::int32>();
    for (int pos = 0; pos < how_many_labels; ++pos) {
      const int label_index = indices_flat(pos);
      const float score = scores_flat(pos);
      top_result.push_back({label_list[label_index], score, 0, 0, 0, 0});
    }
  }
  return top_result;
}

// Analyzes the output of the Inception graph to retrieve the highest scores and
// their positions in the tensor, which correspond to categories.
tensorflow::Status ImageRecognizer::Impl::GetTopLabels(
    const std::vector<tensorflow::Tensor> &outputs, int how_many_labels,
    tensorflow::Tensor *out_indices, tensorflow::Tensor *out_scores) {
  const tensorflow::Tensor &unsorted_scores_tensor = outputs[0];
  auto unsorted_scores_flat = unsorted_scores_tensor.flat<float>();
  std::vector<std::pair<int, float>> scores;
  for (int i = 0; i < unsorted_scores_flat.size(); ++i) {
    scores.push_back(std::pair<int, float>({i, unsorted_scores_flat(i)}));
  }
  std::sort(scores.begin(), scores.end(),
            [](const std::pair<int, float> &left,
               const std::pair<int, float> &right) {
              return left.second > right.second;
            });
  scores.resize(how_many_labels);
  tensorflow::Tensor sorted_indices(tensorflow::DT_INT32, {scores.size()});
  tensorflow::Tensor sorted_scores(tensorflow::DT_FLOAT, {scores.size()});
  for (int i = 0; i < scores.size(); ++i) {
    sorted_indices.flat<int>()(i) = scores[i].first;
    sorted_scores.flat<float>()(i) = scores[i].second;
  }
  *out_indices = sorted_indices;
  *out_scores = sorted_scores;
  return tensorflow::Status::OK();
}

tensorflow::Status ImageRecognizer::Impl::ReadTensorFromImage(
    uint8_t *image_data, int image_width, int image_height, int image_channels,
    std::vector<tensorflow::Tensor> *out_tensors) {

  const int wanted_channels = 3;

  if (image_channels < wanted_channels) {
    return tensorflow::errors::FailedPrecondition(
        "Image needs to have at least ", wanted_channels, " but only has ",
        image_channels);
  }

  if(this->output_layers.size() > 1) { // object detect
    tensorflow::Tensor image_tensor(tensorflow::DT_UINT8,
                                    tensorflow::TensorShape({1, image_height, image_width, image_channels}));
    auto image_tensor_mapped = image_tensor.tensor<uint8_t, 4>();
    uint8_t *out = image_tensor_mapped.data();
    uint32_t total_size = image_height * image_width * image_channels;
    for (uint32_t i = 0, j = 0; i < total_size; ++i) {
      out[i] = image_data[i];
    }
    out_tensors->push_back(image_tensor);
    return tensorflow::Status::OK();
  }

  const int wanted_width = this->input_width;
  const int wanted_height = this->input_height;
  const float input_mean = static_cast<float>(this->input_mean);
  const float input_std = static_cast<float>(this->input_std);


  // In these loops, we convert the eight-bit data in the image into float,
  // resize
  // it using bilinear filtering, and scale it numerically to the float range
  // that
  // the model expects (given by input_mean and input_std).
  tensorflow::Tensor image_tensor(
      tensorflow::DT_FLOAT,
      tensorflow::TensorShape(
          {1, wanted_height, wanted_width, wanted_channels}));
  auto image_tensor_mapped = image_tensor.tensor<float, 4>();
  tensorflow::uint8 *in = image_data;
  float *out = image_tensor_mapped.data();
  const size_t image_rowlen = image_width * image_channels;
  const float width_scale = static_cast<float>(image_width) / wanted_width;
  const float height_scale = static_cast<float>(image_height) / wanted_height;
  for (int y = 0; y < wanted_height; ++y) {
    const float in_y = y * height_scale;
    const int top_y_index = static_cast<int>(floorf(in_y));
    const int bottom_y_index =
        std::min(static_cast<int>(ceilf(in_y)), (image_height - 1));
    const float y_lerp = in_y - top_y_index;
    tensorflow::uint8 *in_top_row = in + (top_y_index * image_rowlen);
    tensorflow::uint8 *in_bottom_row = in + (bottom_y_index * image_rowlen);
    float *out_row = out + (y * wanted_width * wanted_channels);
    for (int x = 0; x < wanted_width; ++x) {
      const float in_x = x * width_scale;
      const int left_x_index = static_cast<int>(floorf(in_x));
      const int right_x_index =
          std::min(static_cast<int>(ceilf(in_x)), (image_width - 1));
      tensorflow::uint8 *in_top_left_pixel =
          in_top_row + (left_x_index * wanted_channels);
      tensorflow::uint8 *in_top_right_pixel =
          in_top_row + (right_x_index * wanted_channels);
      tensorflow::uint8 *in_bottom_left_pixel =
          in_bottom_row + (left_x_index * wanted_channels);
      tensorflow::uint8 *in_bottom_right_pixel =
          in_bottom_row + (right_x_index * wanted_channels);
      const float x_lerp = in_x - left_x_index;
      float *out_pixel = out_row + (x * wanted_channels);
      for (int c = 0; c < wanted_channels; ++c) {
        const float top_left((in_top_left_pixel[c] - input_mean) / input_std);
        const float top_right((in_top_right_pixel[c] - input_mean) / input_std);
        const float bottom_left((in_bottom_left_pixel[c] - input_mean) /
            input_std);
        const float bottom_right((in_bottom_right_pixel[c] - input_mean) /
            input_std);
        const float top = top_left + (top_right - top_left) * x_lerp;
        const float bottom =
            bottom_left + (bottom_right - bottom_left) * x_lerp;
        out_pixel[c] = top + (bottom - top) * y_lerp;
      }
    }
  }

  out_tensors->push_back(image_tensor);
  return tensorflow::Status::OK();
}

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
tensorflow::Status ImageRecognizer::Impl::LoadGraph(
    std::string graph_file_name,
    std::unique_ptr<tensorflow::Session> *session) {
  tensorflow::GraphDef graph_def;
  tensorflow::Status load_graph_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                        graph_file_name, "'");
  }
  // initialize the number of worker threads
  tensorflow::SessionOptions options;
  tensorflow::ConfigProto &config = options.config;
  if (cores > 0) {
    config.set_inter_op_parallelism_threads(cores);
    config.set_intra_op_parallelism_threads(cores);
    config.set_use_per_session_threads(false);
  }

  session->reset(tensorflow::NewSession(options));
  tensorflow::Status session_create_status = (*session)->Create(graph_def);
  if (!session_create_status.ok()) {
    return session_create_status;
  }
  return tensorflow::Status::OK();
}

// Takes a file name, and loads a list of labels from it, one per line, and
// returns a vector of the strings. It pads with empty strings so the length
// of the result is a multiple of 16, because our model expects that.
tensorflow::Status ImageRecognizer::Impl::ReadLabelsFile(
    std::string file_name, std::vector<std::string> *result,
    size_t *found_label_count) {
  std::ifstream file(file_name);
  if (!file) {
    return tensorflow::errors::NotFound("Labels file ", file_name,
                                        " not found.");
  }
  result->clear();
  std::string line;
  while (std::getline(file, line)) {
    result->push_back(line);
  }
  *found_label_count = result->size();
  const int padding = 16;
  while (result->size() % padding) {
    result->emplace_back();
  }
  return tensorflow::Status::OK();
}
// Impl end

// Parameters start
ImageRecognizer::Parameters::Parameters(
    const std::string &graph, const std::string &labels,
    const int32_t input_width, const int32_t input_height,
    const int32_t input_mean, const int32_t input_std,
    const std::string &input_layer,
    const std::vector<std::string> &output_layers, const int32_t cores)
    : graph(graph),
      labels(labels),
      input_width(input_width),
      input_height(input_height),
      input_mean(input_mean),
      input_std(input_std),
      input_layer(input_layer),
      output_layers(output_layers),
      cores(cores) {}

ImageRecognizer::Parameters::~Parameters() {}
// Parameters end

// ImageRecognizer start
ImageRecognizer::ImageRecognizer(const Parameters &params)
    : impl_(std::make_shared<Impl>(params)) {}
ImageRecognizer::~ImageRecognizer() {}

std::vector<ObjectInfo> ImageRecognizer::Recognize(
    uint8_t *image_data, const int image_width, const int image_height,
    const int image_channels) {
  return impl_->Recognize(image_data, image_width, image_height,
                          image_channels);
}
// ImageRecognizer end

std::shared_ptr<ImageRecognizer> CreateImageRecognizer(
    const std::string &graph, const std::string &labels,
    const int32_t input_width, const int32_t input_height,
    const int32_t input_mean, const int32_t input_std,
    const std::string &input_layer,
    const std::vector<std::string> &output_layers,
    const int32_t cores /*= 2*/) {
  ImageRecognizer::Parameters parameters(graph, labels, input_width,
                                         input_height, input_mean, input_std,
                                         input_layer, output_layers, cores);
  return std::make_shared<ImageRecognizer>(parameters);
}
std::ostream &operator<<(std::ostream &os, const ObjectInfo &info) {
  os << info.classes << "[" << info.scores << "]: "
     << "[T: " << info.box_top << ", L: " << info.box_left
     << ", B: " << info.box_bottom << ", R: " << info.box_right << "]";
  return os;
}

}  // recognition
}  // PI