/**
 * @file leaky_relu_layer.hpp
 * @author Abhinav Chanda
 *
 * Definition and implementation of the LeakyReLULayer class.
 */
#ifndef __MLPACK_METHODS_ANN_LAYER_LEAKYRELU_LAYER_HPP
#define __MLPACK_METHODS_ANN_LAYER_LEAKYRELU_LAYER_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the leaky rectifier linearunit layer. A leaky rectifier
 * function has a small slope for x < 0, so that the gradient can flow in 
 * both the directions. The activation function for each neuron in this 
 * layer is the leaky rectifier function, defined by
 *
 * @f{eqnarray*}{
 * f(x) &=& \max(x*alpha, x) \\
 * f'(x) &=& \left\{
 *   \begin{array}{lr}
 *     1 & : x > 0 \\
 *     alpha & : x \le 0
 *   \end{array}
 * \right.
 * @f}
 *
 * For more information, see the following paper:
 *
 * @code
 * @misc{Maas2013,
 *   author = {Andrew L. Maas, Awni Y. Hannun, Andrew Y. Ng},
 *   title = {Rectifier Nonlinearities Improve Neural Network Acoustic Models},
 *   year = {2013}
 * }
 * @endcode
 *
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
class LeakyReLULayer
{
 public:
  /**
   * Create LeakyReLULayer object.
   *
   * @param alpha The leakyness factor of the layer.
   */
  LeakyReLULayer(double alpha) : alpha(alpha)
  { }

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename eT>
  void Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output)
  {
    output = arma::max(input * alpha, input);
  }

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards through f. Using the results from the feed
   * forward pass.
   *
   * @param input The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename DataType>
  void Backward(const DataType& input,
                const DataType& gy,
                DataType& g)
  {
    DataType derivative;
    // Compute the derivative of each element in input.
    for (size_t i = 0; i < input.n_elem; i++)
      derivative(i) = (input(i) > 0) ? 1 : alpha;
    
    g = gy % derivative;
  }

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards trough f. Using the results from the feed
   * forward pass.
   *
   * @param input The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void Backward(const arma::Cube<eT>& input,
                const arma::Mat<eT>& gy,
                arma::Cube<eT>& g)
  {
    // Generate a cube using the backpropagated error matrix.
    arma::Cube<eT> mappedError = arma::zeros<arma::cube>(input.n_rows,
        input.n_cols, input.n_slices);

    for (size_t s = 0, j = 0; s < mappedError.n_slices; s+= gy.n_cols, j++)
    {
      for (size_t i = 0; i < gy.n_cols; i++)
      {
        arma::Col<eT> temp = gy.col(i).subvec(
            j * input.n_rows * input.n_cols,
            (j + 1) * input.n_rows * input.n_cols - 1);

        mappedError.slice(s + i) = arma::Mat<eT>(temp.memptr(),
            input.n_rows, input.n_cols);
      }
    }

    arma::Cube<eT> derivative = input;
    // Compute the derivative of each element in input.
    for (size_t i = 0; i < input.n_elem; i++)
      derivative(i) = (input(i) > 0) ? 1 : alpha;

    g = mappedError % derivative;
  }

  //! Get the alpha.
  double Alpha() const { return alpha; }
  //! Modify the alpha.
  double& Alpha() { return alpha; }

  //! Get the input parameter.
  InputDataType& InputParameter() const { return inputParameter; }
  //! Modify the input parameter.
  InputDataType& InputParameter() { return inputParameter; }

  //! Get the output parameter.
  OutputDataType& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  InputDataType& Delta() const { return delta; }
  //! Modify the delta.
  InputDataType& Delta() { return delta; }

 private:
  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! The leakyness factor for the layer
  double alpha;
}; // class LeakyReLULayer

} // namespace ann
} // namespace mlpack

#endif
