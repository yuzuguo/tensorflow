//
// Created by kerner on 10/17/17.
//
#include <stdio.h>
#include <jpeglib.h>
#include <setjmp.h>
#include <fstream>
#include <vector>
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

// Error handling for JPEG decoding.
void CatchError(j_common_ptr cinfo) {
  (*cinfo->err->output_message)(cinfo);
  jmp_buf *jpeg_jmpbuf = reinterpret_cast<jmp_buf *>(cinfo->client_data);
  jpeg_destroy(cinfo);
  longjmp(*jpeg_jmpbuf, 1);
}

// Decompresses a JPEG file from disk.
tensorflow::Status LoadJpegFile(std::string file_name, std::vector<tensorflow::uint8> *data,
                                int *width, int *height, int *channels) {
  struct jpeg_decompress_struct cinfo;
  FILE *infile;
  JSAMPARRAY buffer;
  int row_stride;

  if ((infile = fopen(file_name.c_str(), "rb")) == NULL) {
    LOG(ERROR) << "Can't open " << file_name;
    return tensorflow::errors::NotFound("JPEG file ", file_name,
                                        " not found");
  }

  struct jpeg_error_mgr jerr;
  jmp_buf jpeg_jmpbuf;  // recovery point in case of error
  cinfo.err = jpeg_std_error(&jerr);
  cinfo.client_data = &jpeg_jmpbuf;
  jerr.error_exit = CatchError;
  if (setjmp(jpeg_jmpbuf)) {
    fclose(infile);
    return tensorflow::errors::Unknown("JPEG decoding failed");
  }

  jpeg_create_decompress(&cinfo);
  jpeg_stdio_src(&cinfo, infile);
  jpeg_read_header(&cinfo, TRUE);
  jpeg_start_decompress(&cinfo);
  *width = cinfo.output_width;
  *height = cinfo.output_height;
  *channels = cinfo.output_components;
  data->resize((*height) * (*width) * (*channels));

  row_stride = cinfo.output_width * cinfo.output_components;
  buffer = (*cinfo.mem->alloc_sarray)
      ((j_common_ptr) &cinfo, JPOOL_IMAGE, row_stride, 1);
  while (cinfo.output_scanline < cinfo.output_height) {
    tensorflow::uint8 *row_address = &((*data)[cinfo.output_scanline * row_stride]);
    jpeg_read_scanlines(&cinfo, buffer, 1);
    memcpy(row_address, buffer[0], row_stride);
  }

  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
  fclose(infile);
  return tensorflow::Status::OK();
}

int main(int argc, char *argv[]) {
  std::string image =
      "tensorflow/contrib/pi_examples/label_image/data/"
          "grace_hopper.jpg";
  std::string graph =
      "tensorflow/contrib/pi_examples/label_image/data/"
          "tensorflow_inception_stripped.pb";
  std::string labels =
      "tensorflow/contrib/pi_examples/label_image/data/"
          "imagenet_comp_graph_label_strings.txt";
  int32_t input_width = 299;
  int32_t input_height = 299;
  int32_t input_mean = 128;
  int32_t input_std = 128;
  std::string input_layer = "Mul";
  std::string output_layer = "softmax";

  std::vector<tensorflow::Flag> flag_list = {
      tensorflow::Flag("image", &image, "image to be processed"),
      tensorflow::Flag("graph", &graph, "graph to be executed"),
      tensorflow::Flag("labels", &labels, "name of file containing labels"),
      tensorflow::Flag("input_width", &input_width, "resize image to this width in pixels"),
      tensorflow::Flag("input_height", &input_height,
           "resize image to this height in pixels"),
      tensorflow::Flag("input_mean", &input_mean, "scale pixel values to this mean"),
      tensorflow::Flag("input_std", &input_std, "scale pixel values to this std deviation"),
      tensorflow::Flag("input_layer", &input_layer, "name of input layer"),
      tensorflow::Flag("output_layer", &output_layer, "name of output layer")
  };
  std::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    std::cerr << usage << std::endl;
    return -1;
  }

  auto imageRecognition = PI::recognition::CreateImageRecognizer(graph,
                                                             labels,
                                                             input_width,
                                                             input_height,
                                                             input_mean,
                                                             input_std,
                                                             input_layer,
                                                             output_layer);

  // Load image
  std::vector<tensorflow::uint8> image_data;
  int image_width;
  int image_height;
  int image_channels;
  std::string image_path = tensorflow::io::JoinPath("", image);
  std::cout << "image path: " << image_path << std::endl;

  tensorflow::Status load_file_status = LoadJpegFile(image_path, &image_data, &image_width,
                                                     &image_height, &image_channels);
  std::cout << "Loaded JPEG: " << image_width << "x" << image_height
            << "x" << image_channels << std::endl;
  if (!load_file_status.ok()) {
    std::cerr << load_file_status << std::endl;
    return -1;
  }

  // run
  tensorflow::uint8 *in = image_data.data();
  auto result = imageRecognition->Recognize(in, image_width, image_height, image_channels);
  for (std::pair<std::string, float> r : result) {
    std::cout << r.first << " : " << r.second << std::endl;
  }
  return 0;
}