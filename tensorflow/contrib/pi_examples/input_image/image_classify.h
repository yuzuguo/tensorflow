//
// Created by kerner on 10/16/17.
//

#ifndef TENSORFLOW_IMAGE_CLASSIFY_H
#define TENSORFLOW_IMAGE_CLASSIFY_H

#include <iostream>
#include <vector>
#include <memory>

namespace PI {
namespace deeplearning {

class ImageClassify {
 public:
  struct Parameters {

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
    // name of input layer, default is Mul
    std::string input_layer;
    // name of output layer, default is softmax
    std::string output_layer;

    Parameters(std::string graph,
               std::string labels,
               int32_t input_width,
               int32_t input_height,
               int32_t input_mean,
               int32_t input_std,
               std::string input_layer,
               std::string output_layer);
    virtual ~Parameters();
  };

  ImageClassify(const Parameters &params);
  virtual ~ImageClassify();

  std::vector<std::pair<std::string, float>> classify(uint8_t *image_data,
                                                      int image_width,
                                                      int image_height,
                                                      int image_channels);

 private:
  // PIMPL
  class Impl;
  std::shared_ptr<Impl> impl_;
};

std::shared_ptr<ImageClassify> CreateImageClassify(std::string graph,
                                                   std::string labels,
                                                   int32_t input_width,
                                                   int32_t input_height,
                                                   int32_t input_mean,
                                                   int32_t input_std,
                                                   std::string input_layer = "Mul",
                                                   std::string output_layer = "softmax");

}
}
#endif //TENSORFLOW_IMAGE_CLASSIFY_H
