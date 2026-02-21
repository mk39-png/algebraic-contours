// Copyright 2023 Adobe Research. All rights reserved.
// To view a copy of the license, visit LICENSE.md.

#include "conic.h"

ConicType
Conic::get_type() const
{
  return m_type;
}

// Assumes row vector points
void
Conic::transform(const Matrix2x2r& rotation, const PlanarPoint& translation)
{
  Matrix3x2r P_rot_coeffs =
    get_numerators() * rotation + get_denominator() * translation;
  // FIXME Remove
  // Matrix3x2r P_coeffs = get_numerators();
  // Eigen::Matrix<double, 3, 1> Px_coeffs = get_numerators().col(0);
  // Eigen::Matrix<double, 3, 1> Py_coeffs = get_numerators().col(1);
  // Eigen::Matrix<double, 3, 1> Q_coeffs = get_denominator();

  // P_rot_coeffs.col(0) = rotation(0, 0) * Px_coeffs + rotation(0, 1) *
  // Py_coeffs; P_rot_coeffs.col(1) = rotation(1, 0) * Px_coeffs + rotation(1,
  // 1) * Py_coeffs; P_rot_coeffs.col(0) += translation(0) * Q_coeffs;
  // P_rot_coeffs.col(1) += translation(1) * Q_coeffs;

  set_numerators(P_rot_coeffs);
}

bool
Conic::is_valid() const
{
  if (m_numerator_coeffs.cols() == 0)
    return false;
  if (m_denominator_coeffs.size() == 0)
    return false;

  return true;
}

// Generated formatted conic string
std::string
Conic::formatted_conic() const
{
  std::stringstream conic_string;
  conic_string << "1/(";
  conic_string << formatted_polynomial<2, 1>(m_denominator_coeffs);
  conic_string << ") [\n  ";
  for (int i = 0; i < m_numerator_coeffs.cols(); ++i) {
    conic_string << formatted_polynomial<2, 1>(m_numerator_coeffs.col(i));
    conic_string << ",\n  ";
  }
  conic_string << "], t in " << m_domain.formatted_interval();

  return conic_string.str();
}

std::ostream&
operator<<(std::ostream& out, const Conic& F)
{
  out << F.formatted_conic();
  return out;
}

// 
// MY HELPERS HERE
// 

std::string conictype_to_str(ConicType conic_type) {
  switch (conic_type) {
    case ConicType::ellipse: return "ELLIPSE";
    case ConicType::hyperbola: return "HYPERBOLA";
    case ConicType::parabola: return "PARABOLA";
    case ConicType::parallel_lines: return "PARALLEL LINES";
    case ConicType::intersecting_lines: return "INTERSECTING LINES";
    case ConicType::line: return "LINE";
    case ConicType::point: return "POINT";
    case ConicType::empty: return "EMPTY";
    case ConicType::plane: return "PLANE";
    case ConicType::error: return "ERROR";
    case ConicType::unknown: return "UNKNOWN";
    default: return "Unreachable Type. ConicType should either be empty, error, or unknown at this point.";
  }
}


std::string serialize_conic(
    const Conic& conic)
{
    std::stringstream output_stream;
    int prec = 17;

    output_stream << "{\n";
    output_stream <<  "  \"degree\": "    << std::to_string(conic.get_degree())    << ",\n";
    output_stream <<  "  \"dimension\": " << std::to_string(conic.get_dimension()) << ",\n";
    output_stream <<  "  \"type\": \"" << conictype_to_str(conic.get_type()) << "\",\n";
    output_stream <<  "  \"numerator_coeffs\": "   << serialize_matrix<2, 2>(conic.get_numerators()) << ",\n";
    output_stream <<  "  \"denominator_coeffs\": " << serialize_matrix<2, 1>(conic.get_denominator()) << ",\n";
    output_stream <<  "  \"domain\": {\n";
    output_stream <<  "    \"t0\": " << std::setprecision(prec) << conic.domain().get_lower_bound() << ",\n";
    output_stream <<  "    \"t1\": " << std::setprecision(prec) << conic.domain().get_upper_bound() << ",\n";
    output_stream <<  "    \"bounded_below\": " << std::boolalpha << conic.domain().is_bounded_below() << ",\n";
    output_stream <<  "    \"bounded_above\": " << std::boolalpha << conic.domain().is_bounded_above() << ",\n";
    output_stream <<  "    \"open_below\": " << std::boolalpha << conic.domain().is_open_below() << ",\n";
    output_stream <<  "    \"open_above\": " << std::boolalpha << conic.domain().is_open_above() << "\n";
    output_stream <<  "  }\n";
    output_stream <<  "}\n";
    
    return output_stream.str();
}

void serialize_vector_conic(    
    std::string filename,
    const std::vector<Conic>& conics)
{   
    // auto dir = Paths::function_dir(filename);
    std::ofstream file(filename, std::ios::out | std::ios::trunc);
    file << "[";

    for (size_t i = 0; i < conics.size(); ++i) {
        auto conic = conics.at(i);
        
        file << serialize_conic(conic);
      
        // Comma and newline to separate Rational Functions
        if (i < conics.size() - 1)  {
            file << ",\n";
        }
    }
    file << "]" << std::endl;
    file.close();
}
