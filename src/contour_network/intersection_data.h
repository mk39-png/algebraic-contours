// Copyright 2023 Adobe Research. All rights reserved.
// To view a copy of the license, visit LICENSE.md.

#pragma once

#include "common.h"

/// Data for intersections
struct IntersectionData
{
  double knot; // Parameter value for the the intersection in the given curve
  size_t intersection_index; // ID of the intersecting curve
  double intersection_knot; // Parameter of the intersection in the intersecting
                            // curve's domain
  bool is_base = false; // True iff the intersection is at the base of the curve
  bool is_tip = false;  // True iff the intersection is at the tip of the curve
  int id;               // Unique identifier for the intersection
  bool is_redundant = false; // Flag for redundant intersections

  /// Check if the knot is the tip of an oriented curve
  ///
  /// @param[in] domain: domain for the curve
  /// @param[in] eps: epsilon tolerance for the check
  void check_if_tip(const Interval& domain, double eps)
  {
    is_tip = (float_equal(domain.get_upper_bound(), knot, eps));
  }

  /// Check if the knot is the base of an oriented curve
  ///
  /// @param[in] domain: domain for the curve
  /// @param[in] eps: epsilon tolerance for the check
  void check_if_base(const Interval& domain, double eps)
  {
    is_base = (float_equal(domain.get_lower_bound(), knot, eps));
  }
};

// Comparator for Intersection data with respect to knot values
struct knot_less_than
{
  inline bool operator()(const IntersectionData& data_1,
                         const IntersectionData& data_2)
  {
    return (data_1.knot < data_2.knot);
  }
};


// 
// 
// TESTING FUNCTION
// 
// 
inline void serialize_intersection_data(std::string filename, const std::vector<std::vector<IntersectionData>>& contour_intersections) {
  std::ofstream output_file_0(filename, std::ios::out | std::ios::trunc);
  output_file_0 << "[\n";

  // Now for every element..
  for (size_t i = 0; i < contour_intersections.size(); ++i) {
    output_file_0 << "  [\n"; // start inner array
    for (size_t j = 0; j < contour_intersections.at(i).size(); ++j) {
      output_file_0 << "    {\n";
      output_file_0 << "      \"knot\": " << std::setprecision(17) << contour_intersections.at(i).at(j).knot << ",\n";
      output_file_0 << "      \"intersection_index\": " + std::to_string(contour_intersections.at(i).at(j).intersection_index) + ",\n";
      output_file_0 << "      \"intersection_knot\": " << std::setprecision(17) << contour_intersections.at(i).at(j).intersection_knot << ",\n";
      output_file_0 << "      \"is_base\": " << std::boolalpha << contour_intersections.at(i).at(j).is_base << ",\n";
      output_file_0 << "      \"is_tip\": " << std::boolalpha << contour_intersections.at(i).at(j).is_tip << ",\n";
      output_file_0 << "      \"id\": " + std::to_string(contour_intersections.at(i).at(j).id) + ",\n";
      output_file_0 << "      \"is_redundant\": " << std::boolalpha << contour_intersections.at(i).at(j).is_redundant << "\n";
      output_file_0 << "    }";
      if (j + 1 != contour_intersections.at(i).size()) output_file_0 << ",";
      output_file_0 << "\n";
    }
    output_file_0 << "  ]";
    if (i + 1 != contour_intersections.size()) output_file_0 << ",";
    output_file_0 << "\n";
  }
  output_file_0 << "]\n";
  output_file_0.close();

}