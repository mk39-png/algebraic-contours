// Copyright 2023 Adobe Research. All rights reserved.
// To view a copy of the license, visit LICENSE.md.

#pragma once

#include "polyscope/curve_network.h"

#include <string>
#include <sstream>
#include "common.h"
#include "interval.h"
#include "polynomial_function.h"

struct CurveDiscretizationParameters
{
  int num_samples = 5;
  int num_tangents_per_segment = 5;
};

/// @brief  Representation of a vector valued rational function f: R -> R^n.
template<size_t degree, size_t dimension>
class RationalFunction
{
public:
  // ************
  // Constructors
  // ************

  /// Default constructor for 0 function
  RationalFunction()
  {
    // Default numerator to constant 0 in R^n
    m_numerator_coeffs.setZero();
    m_denominator_coeffs.setZero();
    m_denominator_coeffs[0] = 1.0;
    m_domain.reset_bounds();

    assert(is_valid());
  }

  /// Constructor for a vector polynomial function.
  ///
  /// @param[in] numerator_coeffs: coefficients of the polynomial functions
  RationalFunction(
    const Eigen::Matrix<double, degree + 1, dimension>& numerator_coeffs)
    : m_numerator_coeffs(numerator_coeffs)
  {
    m_denominator_coeffs.setZero();
    m_denominator_coeffs[0] = 1.0;
    m_domain.reset_bounds();

    assert(is_valid());
  }

  /// General constructor over entire real line.
  ///
  /// @param[in] numerator_coeffs: coefficients of the numerator polynomials
  /// @param[in] denominator_coeffs: coefficients of the denominator polynomial
  RationalFunction(
    const Eigen::Matrix<double, degree + 1, dimension>& numerator_coeffs,
    const Eigen::Matrix<double, degree + 1, 1>& denominator_coeffs)
    : m_numerator_coeffs(numerator_coeffs)
    , m_denominator_coeffs(denominator_coeffs)
  {
    m_domain.reset_bounds();

    assert(is_valid());
  }

  /// General constructor over given interval.
  ///
  /// @param[in] numerator_coeffs: coefficients of the numerator polynomials
  /// @param[in] denominator_coeffs: coefficients of the denominator polynomial
  /// @param[in] domain: domain interval for the mapping
  RationalFunction(
    const Eigen::Matrix<double, degree + 1, dimension>& numerator_coeffs,
    const Eigen::Matrix<double, degree + 1, 1>& denominator_coeffs,
    const Interval& domain)
    : m_numerator_coeffs(numerator_coeffs)
    , m_denominator_coeffs(denominator_coeffs)
    , m_domain(domain)
  {
    assert(is_valid());
  }

  // *******
  // Methods
  // *******

  /// Compute the degree of the polynomial mapping as the max of the degrees
  /// of the numerator and denominator degrees.
  ///
  /// @return degree of the rational mapping
  int get_degree() const { return degree; }

  /// Compute the dimension of the rational mapping.
  ///
  /// @return dimension of the rational mapping
  int get_dimension() const { return dimension; }

  /// Compute the derivative of the rational function, which is also a rational
  /// function, using the quotient rule.
  ///
  /// @param[out] derivative: derivative rational function
  void compute_derivative(
    RationalFunction<2 * degree, dimension>& derivative) const
  {
    // Compute the derivatives of the numerator and denominator polynomials
    spdlog::trace("Taking derivative of rational function");
    spdlog::trace("Numerator:\n{}", m_numerator_coeffs);
    spdlog::trace("Denominator:\n{}", m_denominator_coeffs);
    Eigen::Matrix<double, degree, dimension> numerator_deriv_coeffs;
    compute_polynomial_mapping_derivative<degree, dimension>(
      m_numerator_coeffs, numerator_deriv_coeffs);
    Eigen::Matrix<double, degree, 1> denominator_deriv_coeffs;
    compute_polynomial_mapping_derivative<degree, 1>(m_denominator_coeffs,
                                                     denominator_deriv_coeffs);
    spdlog::trace("Numerator derivative:\n{}", numerator_deriv_coeffs);
    spdlog::trace("Denominator derivative:\n{}", denominator_deriv_coeffs);

    // FIXME 0 degree case

    // Compute the derivative numerator and denominator from the quotient rule
    Eigen::Matrix<double, 2 * degree, dimension> term_0, term_1;
    compute_polynomial_mapping_scalar_product<degree, degree - 1, dimension>(
      m_denominator_coeffs, numerator_deriv_coeffs, term_0);
    compute_polynomial_mapping_scalar_product<degree - 1, degree, dimension>(
      denominator_deriv_coeffs, m_numerator_coeffs, term_1);
    spdlog::trace("First term:\n{}", term_0);
    spdlog::trace("Second term:\n{}", term_1);
    Eigen::Matrix<double, 2 * degree + 1, dimension> num_coeffs;
    num_coeffs.setZero();
    num_coeffs.block(0, 0, 2 * degree, dimension) = term_0 - term_1;
    Eigen::Matrix<double, 2 * degree + 1, 1> denom_coeffs;
    compute_polynomial_mapping_product<degree, degree, 1>(
      m_denominator_coeffs, m_denominator_coeffs, denom_coeffs);

    // Build the derivative
    derivative = RationalFunction<2 * degree, dimension>(
      num_coeffs, denom_coeffs, m_domain);
  }

  /// @brief Compose the rational mapping f: R -> R^n with a one form to obtain
  /// a rational scalar.
  ///
  /// @param[in] one_form: One form w: R^n -> R to apply to the rational mapping
  /// @param[out] scalar_function: composed scalar rational function
  void apply_one_form(const Eigen::Matrix<double, dimension, 1>& one_form,
                      RationalFunction<degree, 1> scalar_function) const
  {
    // Compute the scalar polynomial numerator coefficients
    Eigen::Matrix<double, degree + 1, 1> numerator_coeffs;
    numerator_coeffs = m_numerator_coeffs * one_form;

    // Create a scalar rational function with the same domain and denominator
    // but the new numerator
    scalar_function = RationalFunction<degree, 1>(
      numerator_coeffs, m_denominator_coeffs, m_domain);
  }

  /// @brief Split the rational function into two rational function at some knot
  /// in the domain.
  ///
  /// @param[in] knot: point in the domain to split the function at
  /// @param[out] lower_segment: rational function with lower domain
  /// @param[out] upper_segment: rational function with upper domain
  void split_at_knot(double knot,
                     RationalFunction<degree, dimension>& lower_segment,
                     RationalFunction<degree, dimension>& upper_segment)
  {
    // Build lower segment
    double t0 = m_domain.get_lower_bound();
    assert(t0 <= knot);
    Interval lower_domain(t0, knot);
    lower_segment = RationalFunction<degree, dimension>(
      m_numerator_coeffs, m_denominator_coeffs, lower_domain);

    // Build upper segment
    double t1 = m_domain.get_upper_bound();
    assert(knot <= t1);
    Interval upper_domain(knot, t1);
    upper_segment = RationalFunction<degree, dimension>(
      m_numerator_coeffs, m_denominator_coeffs, upper_domain);
  }

  /// @brief Sample points in the rational function.
  ///
  /// @param[in] num_points: number of points to sample
  /// @param[out] points: vector of sampled points
  void sample_points(
    int num_points,
    std::vector<Eigen::Matrix<double, 1, dimension>>& points) const
  {
    // Get sample of the domain
    std::vector<double> t_samples = m_domain.sample_points(num_points);

    // Evaluate the function at the sampled domain points
    points.resize(num_points);
    for (int i = 0; i < num_points; ++i) {
      evaluate(t_samples[i], points[i]);
    }
  }

  /// @brief Get the point at the start of the rational mapping curve.
  ///
  /// @return curve start point in R^n
  Eigen::Matrix<double, 1, dimension> start_point() const
  {
    // Return the default constructor point if the domain is not bounded below
    if (!m_domain.is_bounded_below())
      return Eigen::Matrix<double, 1, dimension>();

    double t0 = m_domain.get_lower_bound();
    return evaluate(t0);
  }

  /// @brief Get the point of the rational mapping curve sampled at the midpoint
  /// of the domain interval (or some interior point in an unbounded interval).
  ///
  /// @return curve mid point in R^n
  Eigen::Matrix<double, 1, dimension> mid_point() const
  {
    // Return the default constructor point if the domain is not bounded below
    if (!m_domain.is_bounded_below())
      return Eigen::Matrix<double, 1, dimension>();
    if (!m_domain.is_bounded_above())
      return Eigen::Matrix<double, 1, dimension>();

    double t0 = m_domain.get_lower_bound();
    double t1 = m_domain.get_upper_bound();
    return evaluate((t0 + t1) / 2.0);
  }

  /// @brief Get the point at the end of the rational mapping curve.
  ///
  /// @return curve end point in R^n
  Eigen::Matrix<double, 1, dimension> end_point() const
  {
    // Return the default constructor point if the domain is not bounded below
    if (!m_domain.is_bounded_above())
      return Eigen::Matrix<double, 1, dimension>();

    double t1 = m_domain.get_upper_bound();
    return evaluate(t1);
  }

  /// @brief Evaluate the function at a domain coordinate
  ///
  /// @param[in] t: coordinate
  /// @param[out] point: rational function evaluated at coordinate t
  void evaluate(double t, Eigen::Matrix<double, 1, dimension>& point) const
  {
    Eigen::Matrix<double, 1, dimension> Pt;
    Eigen::Matrix<double, 1, 1> Qt;
    evaluate_polynomial_mapping<degree, dimension>(m_numerator_coeffs, t, Pt);
    evaluate_polynomial_mapping<degree, 1>(m_denominator_coeffs, t, Qt);

    point = Pt / Qt[0];
  }

  /// @brief Evaluate the function at an normalized parameter in [0, 1]
  ///
  /// @param[in] t: normalized coordinate
  /// @param[out] point: rational function evaluated at normalized coordinate t
  void evaluate_normalized_coordinate(
    double t,
    Eigen::Matrix<double, 1, dimension>& point) const
  {
    // Check if domain is bounded
    if (!domain().is_bounded_below())
      return;
    if (!domain().is_bounded_above())
      return;

    // Linearly interpolate the coordinate
    double t0 = domain().get_lower_bound();
    double t1 = domain().get_upper_bound();
    double s = interval_lerp(0.0, 1.0, t0, t1, t);

    // Evaluate at the given domain coordinate
    evaluate(s, point);
  }

  /// @brief Determine if a point is in the domain of the rational mapping.
  ///
  /// @return true iff t is in the domain
  bool is_in_domain(double t) const { return m_domain.contains(t); }

  /// @brief Determine if a point is in the interior of the domain of the
  /// rational mapping.
  ///
  /// @return true iff t is in the domain interior
  bool is_in_domain_interior(double t) const
  {
    return m_domain.is_in_interior(t);
  }

  /// Discretize the given rational curve as a polyline curve network
  ///
  /// @param[in] curve_disc_params: parameters for the curve discretization
  /// @param[out] points: points of the curve network
  /// @param[out] polyline: polyline indices of the curve network
  void discretize(const CurveDiscretizationParameters& curve_disc_params,
                  std::vector<Eigen::Matrix<double, 1, dimension>>& points,
                  std::vector<int>& polyline) const
  {
    // Write curves
    int num_samples = curve_disc_params.num_samples;
    sample_points(num_samples, points);

    // Build polyline for the given curve
    polyline.resize(points.size());
    for (size_t l = 0; l < points.size(); ++l) {
      polyline[l] = l;
    }
  }

  /// @brief Add the rational function curve to the polyscope viewer.
  ///
  /// Note that this method only works for rational space curves.
  ///
  /// @param[in] curve_name: name to assign the curve in the viewer
  void add_curve_to_viewer(
    std::string curve_name = "rational_function_curve") const
  {
    if (dimension != 3) {
      spdlog::error("Cannot view nonspatial curve");
      return;
    }

    // Generate curve discretization
    CurveDiscretizationParameters curve_disc_params;
    std::vector<Eigen::Matrix<double, 1, dimension>> points;
    std::vector<int> polyline;
    discretize(curve_disc_params, points, polyline);

    // Add curve mesh
    MatrixXr points_mat = convert_nested_vector_to_matrix(points);
    std::vector<std::vector<int>> polylines = { polyline };
    std::vector<std::array<int, 2>> edges =
      convert_polylines_to_edges(polylines);
    polyscope::init();
    polyscope::registerCurveNetwork(curve_name, points_mat, edges);
    polyscope::getCurveNetwork(curve_name)->setRadius(0.0025);
  }

  /// @brief Compute the derivative at domain point t with finite differences
  /// with finite difference step size h.
  ///
  /// This method should only be used for validation; the derivative method is
  /// exact.
  ///
  /// @param[in] t: point to evaluate the derivative at
  /// @param[in] h: finite difference step size
  /// @return finite difference derivative
  std::vector<Eigen::Matrix<double, 1, dimension>> finite_difference_derivative(
    double t,
    double h = 1e-3) const
  {
    std::vector<Eigen::Matrix<double, 1, dimension>> F_plus, F_minus;
    evaluate(t + h, F_plus);
    evaluate(t - h, F_minus);
    return (F_plus - F_minus) / (2 * h);
  }

  // *******************
  // Getters and setters
  // *******************
  void set_numerators(
    const Eigen::Matrix<double, degree + 1, dimension>& numerator)
  {
    m_numerator_coeffs = numerator;
  }
  void set_denominator(const Eigen::Matrix<double, degree + 1, 1>& denominator)
  {
    m_denominator_coeffs = denominator;
  }
  Eigen::Matrix<double, degree + 1, dimension> const& get_numerators() const
  {
    return m_numerator_coeffs;
  }
  Eigen::Matrix<double, degree + 1, 1> const& get_denominator() const
  {
    return m_denominator_coeffs;
  }

  // TODO: Fix direct exposure of domain
  Interval& domain() { return m_domain; }
  Interval const& domain() const { return m_domain; }

  // ********************
  // Overloaded operators
  // ********************

  /// @brief Write a human readable formatting of the rational function to a
  /// stream.
  ///
  /// @param[in] out: output stream
  /// @param[in] F: rational function to write
  /// @return output stream for chaining
  template<size_t output_degree, size_t output_dimension>
  friend std::ostream& operator<<(
    std::ostream& out,
    const RationalFunction<output_degree, output_dimension>& F);

  /// @brief Evaluate the rational mapping at domain point t.
  ///
  /// @param[in] t: domain point to evaluate at
  /// @return evaluated point
  Eigen::Matrix<double, 1, dimension> operator()(double t) const
  {
    return evaluate(t);
  }

  // TODO Implement
  // RationalFunction operator + (const RationalFunction &obj) const;
  // RationalFunction operator - () const;
  // RationalFunction operator - (const RationalFunction &obj) const;

private:
  friend class Conic;

  bool is_valid() const
  {
    if (m_numerator_coeffs.cols() == 0)
      return false;
    if (m_denominator_coeffs.size() == 0)
      return false;

    return true;
  }

  // ******************************
  // Helper functions for operators
  // ******************************

  Eigen::Matrix<double, 1, dimension> evaluate(double t) const
  {
    Eigen::Matrix<double, 1, dimension> Pt;
    Eigen::Matrix<double, 1, 1> Qt;
    evaluate_polynomial_mapping<degree, dimension>(m_numerator_coeffs, t, Pt);
    evaluate_polynomial_mapping<degree, 1>(m_denominator_coeffs, t, Qt);

    return Pt / Qt[0];
  }

  std::string formatted_rational_function() const
  {
    std::stringstream rational_function_string;
    rational_function_string << "1/(";
    rational_function_string
      << formatted_polynomial<degree, 1>(m_denominator_coeffs, 17);
    rational_function_string << ") [\n  ";
    for (int i = 0; i < m_numerator_coeffs.cols(); ++i) {
      rational_function_string
        << formatted_polynomial<degree, 1>(m_numerator_coeffs.col(i), 17);
      rational_function_string << ",\n  ";
    }
    rational_function_string << "], t in " << m_domain.formatted_interval();

    return rational_function_string.str();
  }

  // ****************
  // Member variables
  // ****************

  Eigen::Matrix<double, degree + 1, dimension> m_numerator_coeffs;
  Eigen::Matrix<double, degree + 1, 1> m_denominator_coeffs;
  Interval m_domain;
};

template<size_t degree, size_t dimension>
std::ostream&
operator<<(std::ostream& out, const RationalFunction<degree, dimension>& F)
{
  out << F.formatted_rational_function();
  return out;
}



// 
// MY HELPERS
// 

// Returns string of matrix, without any formatting. just a single line of it.
// This is for formatting numerators and denominator of a Rational Function
// REMINDER: + 1 because there's the constant and then the variable terms inside the rational function
// FIXME problem with the function below... causing major compilation errors
// This prints row-major order, buit in a very werid way...
// For some reason, using .col rather than direct indexing... or something like that...
// Why? IDK, I just thought of making this like this for this reason.
// Is it cumbersome? Very.
template <size_t degree, size_t dimension>
std::string serialize_matrix(const Eigen::Matrix<double, degree + 1, dimension> coeffs) {
    // TODO: check if this is the right order.
    std::stringstream output_stream;
    output_stream << "[";
    int prec = 17;

    // this outer loop below is equivalent to iteration through __dimension__ (e.g. 3)
    for (long int i = 0; i < coeffs.cols(); ++i) { 
        output_stream << "[";

        // this loop below is equivalent as iterating through __degree__ (e.g. 4)
        for (long int j = 0; j < coeffs.rows(); ++j) {
            // Need to print out to a certain precision.
            // FIXME: is the below even correct? Like, iterating through a specific row...
            output_stream << std::setprecision(prec) << coeffs.col(i)[j];

            // Ensuring not putting a comma on the last element in the row as to avoid trailing commas
            if (j < coeffs.rows() - 1) {
              output_stream << ",";
            }
        }
        output_stream << "]";


        // We do not want a comma on the last inner array though.
        // So, make sure that we are not placing comma for 1D matrices.
        // And also making sure not placing comma at the end of the last inner array.
        if ((coeffs.rows() > 1) && (i < coeffs.cols() - 1)) {
            output_stream << ", ";
        }
    }
    output_stream << "]";


    return output_stream.str();
}


// Turns a rational function into JSON file.
template <size_t degree, size_t dimension>
std::string serialize_rational_function(
    const RationalFunction<degree, dimension>& rational_function)
{
    std::stringstream output_stream;
    int prec = 17;

    output_stream << "{\n";
    output_stream <<  "  \"degree\": "    << std::to_string(rational_function.get_degree())    << ",\n";
    output_stream <<  "  \"dimension\": " << std::to_string(rational_function.get_dimension()) << ",\n";
    output_stream <<  "  \"numerator_coeffs\": "   << serialize_matrix<degree, dimension>(rational_function.get_numerators()) << ",\n";
    output_stream <<  "  \"denominator_coeffs\": " << serialize_matrix<degree, 1>(rational_function.get_denominator()) << ",\n";
    output_stream <<  "  \"domain\": {\n";
    output_stream <<  "    \"t0\": " << std::setprecision(prec) << rational_function.domain().get_lower_bound() << ",\n";
    output_stream <<  "    \"t1\": " << std::setprecision(prec) << rational_function.domain().get_upper_bound() << ",\n";
    output_stream <<  "    \"bounded_below\": " << std::boolalpha << rational_function.domain().is_bounded_below() << ",\n";
    output_stream <<  "    \"bounded_above\": " << std::boolalpha << rational_function.domain().is_bounded_above() << ",\n";
    output_stream <<  "    \"open_below\": " << std::boolalpha << rational_function.domain().is_open_below() << ",\n";
    output_stream <<  "    \"open_above\": " << std::boolalpha << rational_function.domain().is_open_above() << "\n";
    output_stream <<  "  }\n";
    output_stream <<  "}\n";
    
    return output_stream.str();
}



template <size_t degree, size_t dimension>
void serialize_vector_rational_function(    
    std::string filename,
    const std::vector<RationalFunction<degree, dimension>>& rational_functions)
{
    std::ofstream file(filename, std::ios::out | std::ios::trunc);
    file << "[";

    for (size_t i = 0; i < rational_functions.size(); ++i) {
        auto rational_function = rational_functions.at(i);
        
        file << serialize_rational_function<degree, dimension>(rational_function);
      
        // Comma and newline to separate Rational Functions
        if (i < rational_functions.size() - 1)  {
            file << ",\n";
        }
    }
    file << "]" << std::endl;
    file.close();
}
