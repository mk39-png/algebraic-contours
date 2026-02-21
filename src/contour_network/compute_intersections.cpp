// Copyright 2023 Adobe Research. All rights reserved.
// To view a copy of the license, visit LICENSE.md.

#include "compute_intersections.h"

#include "polynomial_function.h"
#include <chrono>

// Map from the uniform domain [0, 1] to the planar curve domain
double
convert_spline_to_planar_curve_parameter(
  const RationalFunction<4, 2>& planar_curve,
  double t_spline,
  double epsilon = 0)
{
  double t_min = planar_curve.domain().get_lower_bound() + epsilon;
  double t_max = planar_curve.domain().get_upper_bound() - epsilon;
  return interval_lerp(0, 1, t_max, t_min, t_spline);
}

// Compute planar curve intersections with Bezier clipping
void
compute_bezier_clipping_planar_curve_intersections(
  const RationalFunction<4, 2>& first_planar_curve,
  const RationalFunction<4, 2>& second_planar_curve,
  std::vector<std::pair<double, double>>& intersection_points,
  const Eigen::Matrix<double, 5, 3>& first_bezier_control_points,
  const Eigen::Matrix<double, 5, 3>& second_bezier_control_points,
  double epsilon = 0)
{
  intersection_points.clear();

  // 
  // TESTING:
  // 
  static size_t counter = 0;
  std::string filepath = "spot_control/contour_network/compute_intersections/compute_bezier_clipping_planar_curve_intersections/";
  serialize_vector_rational_function<4, 2>(filepath+"first_planar_curve/"+std::to_string(counter)+".json", {first_planar_curve});
  serialize_vector_rational_function<4, 2>(filepath+"second_planar_curve/"+std::to_string(counter)+".json", {second_planar_curve});
  serialize_eigen_matrix_d(filepath+"first_bezier_control_points/"+std::to_string(counter)+".csv", first_bezier_control_points);
  serialize_eigen_matrix_d(filepath+"second_bezier_control_points/"+std::to_string(counter)+".csv", second_bezier_control_points);



  // FIXME This is inefficient
  Point curve1[5], curve2[5];
  for (int i = 0; i < 5; i++) {
    curve1[i] = first_bezier_control_points.row(i);
    curve2[i] = second_bezier_control_points.row(i);
  }

  // FIXME
  if (!check_split_criteria(curve1)) {
    // std::cout << "potential self intersection p1!" << std::endl;
  }

  if (!check_split_criteria(curve2)) {
    // std::cout << "potential self intersection p2!" << std::endl;
  }

  std::vector<std::pair<double, double>> intersection_param_inkscope;
  std::vector<Point> P1(5);
  std::vector<Point> P2(5);
  for (int i = 0; i < 5; i++) {
    P1[i] = first_bezier_control_points.row(i);
    P2[i] = second_bezier_control_points.row(i);
  }

  // // std::cout << "in" << std::endl;
  find_intersections_bezier_clipping(
    intersection_param_inkscope,
    P1,
    P2,
    FIND_INTERSECTIONS_BEZIER_CLIPPING_PRECISION); // 1e-7
  // // std::cout << intersection_param_inkscope.size() << std::endl;

  for (size_t i = 0; i < intersection_param_inkscope.size(); i++) {
    double t_spline = intersection_param_inkscope[i].first;
    double s_spline = intersection_param_inkscope[i].second;
    double t = convert_spline_to_planar_curve_parameter(
      first_planar_curve, t_spline, epsilon);
    double s = convert_spline_to_planar_curve_parameter(
      second_planar_curve, s_spline, epsilon);
    intersection_points.push_back(std::make_pair(t, s));
  }

  // 
  // TESTING
  // 
  serialize_vector_pair_index(filepath+"intersection_points/"+std::to_string(counter)+".csv", intersection_points);
  counter++;

}

// Prune curve intersection points to the proper domains
void
prune_intersection_points(
  const RationalFunction<4, 2>& first_planar_curve,
  const RationalFunction<4, 2>& second_planar_curve,
  const std::vector<std::pair<double, double>>& intersection_points,
  std::vector<double>& first_curve_intersections,
  std::vector<double>& second_curve_intersections)
{
  // 
  // TESTING
  // 
  static size_t counter = 0;
  std::string filepath = "spot_control/contour_network/compute_intersections/prune_intersection_points/";
  // HACK: wrapping in vector
  serialize_vector_rational_function<4, 2>(
    filepath+"first_planar_curve/"+std::to_string(counter)+".json", {first_planar_curve});
  serialize_vector_rational_function<4, 2>(
    filepath+"second_planar_curve/"+std::to_string(counter)+".json", {second_planar_curve});
  serialize_vector_pair_index(
    filepath+"intersection_points/"+std::to_string(counter)+".csv", intersection_points);
  serialize_vector_int(
    filepath+"first_curve_intersections_in/"+std::to_string(counter)+".csv", first_curve_intersections);
  serialize_vector_int(
    filepath+"second_curve_intersections_in/"+std::to_string(counter)+".csv", second_curve_intersections);


  for (size_t i = 0; i < intersection_points.size(); ++i) {
    double t = intersection_points[i].first;
    double s = intersection_points[i].second;

    // Trim points entirely out of domain of one of the two curves
    if (!first_planar_curve.is_in_domain_interior(t))
      continue;
    if (!second_planar_curve.is_in_domain_interior(s))
      continue;

    // Add points otherwise
    first_curve_intersections.push_back(t);
    second_curve_intersections.push_back(s);
  }

  // 
  // TESTING
  // 
  serialize_vector_int(
    filepath+"first_curve_intersections_out/"+std::to_string(counter)+".csv", first_curve_intersections);
  serialize_vector_int(
    filepath+"second_curve_intersections_out/"+std::to_string(counter)+".csv", second_curve_intersections);
  counter++;

}


void serialize_intersection_stats(std::string filename, const IntersectionStats& intersection_stats) {
  // turns to JSON file
  std::ofstream file(filename, std::ios::out | std::ios::trunc);
  file << "{\n";
  file <<  "  " << "\"num_intersection_tests\": " << intersection_stats.num_intersection_tests << ",\n";
  file <<  "  " << "\"num_bezier_nonoverlaps\": " << intersection_stats.num_bezier_nonoverlaps << ",\n";
  file <<  "  " << "\"bounding_box_call\": " << intersection_stats.bounding_box_call << ",\n";
  file <<  "  " << "\"intersection_call\": " << intersection_stats.intersection_call << "\n";
  file << "}" << std::endl;
}

void serialize_intersection_params(std::string filename, const IntersectionParameters& intersect_params) {
  // turns to JSON file
  std::ofstream file(filename, std::ios::out | std::ios::trunc);
  file << "{\n";
  file <<  "  " << "\"use_heuristics\": " << std::boolalpha << intersect_params.use_heuristics << ",\n";
  file <<  "  " << "\"trim_amount\": " << std::setprecision(17) << intersect_params.trim_amount << "\n";
  file << "}" << std::endl;
}



void
compute_planar_curve_intersections(
  const RationalFunction<4, 2>& first_planar_curve,
  const RationalFunction<4, 2>& second_planar_curve,
  const IntersectionParameters& intersect_params,
  std::vector<double>& first_curve_intersections,
  std::vector<double>& second_curve_intersections,
  IntersectionStats& intersection_stats,
  const std::pair<PlanarPoint, PlanarPoint>& first_bounding_box,
  const std::pair<PlanarPoint, PlanarPoint>& second_bounding_box,
  const Eigen::Matrix<double, 5, 3>& first_bezier_control_points,
  const Eigen::Matrix<double, 5, 3>& second_bezier_control_points)
{
  // // TODO: serialize everything (include the intersection stats... oh great)
  // static size_t counter = 0;
  // std::string filepath = "spot_control/contour_network/compute_intersections/compute_planar_curve_intersections/";
  // serialize_vector_rational_function(
  //   filepath+"first_planar_curve/"+std::to_string(counter)+".json", 
  //   std::vector{first_planar_curve});
  // serialize_vector_rational_function(
  //   filepath+"second_planar_curve/"+std::to_string(counter)+".json",
  //   std::vector{second_planar_curve});
  // // NOTE: only need to serialize once
  // serialize_intersection_params(filepath+"intersect_params.json", intersect_params);

  // serialize_intersection_stats(
  //   filepath+"intersection_stats_in/"+std::to_string(counter)+".json",
  //   intersection_stats);
  // serialize_vector_pair_planarpoint(
  //   filepath+"first_bounding_box/"+std::to_string(counter)+".csv",
  //   std::vector{first_bounding_box});
  // serialize_vector_pair_planarpoint(
  //   filepath+"second_bounding_box/"+std::to_string(counter)+".csv",
  //   std::vector{second_bounding_box});
  // serialize_eigen_matrix_d(
  //   filepath+"first_bezier_control_points/"+std::to_string(counter)+".csv",
  //   first_bezier_control_points);
  // serialize_eigen_matrix_d(
  //   filepath+"second_bezier_control_points/"+std::to_string(counter)+".csv",
  //   second_bezier_control_points);



  intersection_stats.num_intersection_tests++;
  auto t1 = std::chrono::high_resolution_clock::now();
  spdlog::trace("Finding intersections for {} and {}",
                first_planar_curve,
                second_planar_curve);

  intersection_stats.bounding_box_call++;

  if (intersect_params.use_heuristics &&
      are_nonintersecting_by_heuristic(
        first_bounding_box, second_bounding_box, intersection_stats)) {
    return;
  }

  intersection_stats.intersection_call++;

  // Compute intersection points by Bezier clipping
  std::vector<std::pair<double, double>> intersection_points;
  try {
    compute_bezier_clipping_planar_curve_intersections(
      first_planar_curve,
      second_planar_curve,
      intersection_points,
      first_bezier_control_points,
      second_bezier_control_points);
  } catch (std::runtime_error&) {
    intersection_points.clear();
    spdlog::error("Failed to find intersection points");
  }

  // Prune the computed intersections to ensure they are in the correct domain
  prune_intersection_points(first_planar_curve,
                            second_planar_curve,
                            intersection_points,
                            first_curve_intersections,
                            second_curve_intersections);

  // Record the time spent finding intersections
  auto t2 = std::chrono::high_resolution_clock::now();
  double total_time =
    std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  spdlog::trace("Finding intersections took {} ms", total_time);


  // // TESTING: serialize results
  // serialize_vector_int(
  //   filepath+"first_curve_intersections/"+std::to_string(counter)+".csv",
  //   first_curve_intersections);
  // serialize_vector_int(
  //   filepath+"second_curve_intersections/"+std::to_string(counter)+".csv",
  //   second_curve_intersections);
  // serialize_intersection_stats(
  //   filepath+"intersection_stats_out/"+std::to_string(counter)+".json",
  //   intersection_stats);
  // counter++;
}

void
compute_planar_curve_intersections(
  const RationalFunction<4, 2>& first_planar_curve,
  const RationalFunction<4, 2>& second_planar_curve,
  const IntersectionParameters& intersect_params,
  std::vector<double>& first_curve_intersections,
  std::vector<double>& second_curve_intersections,
  IntersectionStats& intersection_stats)
{
  // Compute bounding boxes
  PlanarPoint lower_left_point, upper_right_point;
  std::pair<PlanarPoint, PlanarPoint> first_bounding_box;
  std::pair<PlanarPoint, PlanarPoint> second_bounding_box;
  compute_bezier_bounding_box(
    first_planar_curve, lower_left_point, upper_right_point);
  first_bounding_box = std::make_pair(lower_left_point, upper_right_point);
  compute_bezier_bounding_box(
    second_planar_curve, lower_left_point, upper_right_point);
  second_bounding_box = std::make_pair(lower_left_point, upper_right_point);

  // Compute Bezier points
  Eigen::Matrix<double, 5, 3> first_bezier_control_points;
  Eigen::Matrix<double, 5, 3> second_bezier_control_points;
  compute_homogeneous_bezier_points_over_interval(
    first_planar_curve,
    first_planar_curve.domain().get_lower_bound(),
    first_planar_curve.domain().get_upper_bound(),
    first_bezier_control_points);
  compute_homogeneous_bezier_points_over_interval(
    second_planar_curve,
    second_planar_curve.domain().get_lower_bound(),
    second_planar_curve.domain().get_upper_bound(),
    second_bezier_control_points);

  // Compute intersections with computed bounding boxes and Bezier points
  compute_planar_curve_intersections(first_planar_curve,
                                     second_planar_curve,
                                     intersect_params,
                                     first_curve_intersections,
                                     second_curve_intersections,
                                     intersection_stats,
                                     first_bounding_box,
                                     second_bounding_box,
                                     first_bezier_control_points,
                                     second_bezier_control_points);
}

void
compute_intersections(
  const std::vector<RationalFunction<4, 2>>& image_segments,
  const IntersectionParameters& intersect_params,
  std::vector<std::vector<double>>& intersections,
  std::vector<std::vector<size_t>>& intersection_indices,
  std::vector<std::vector<IntersectionData>>& contour_intersections,
  int& num_intersections,
  long long& intersection_call)
{
  // 
  // TESTING: Serializing parameters as well for fast testing.
  // 
  std::string filepath_0 = "spot_control/contour_network/compute_intersections/compute_intersections/";
  serialize_vector_rational_function<4, 2>(filepath_0+"image_segments.json", image_segments);

  std::ofstream output_file_in_0(filepath_0+"num_intersections_in.txt", std::ios::out | std::ios::trunc);
  output_file_in_0 << std::to_string(num_intersections) << std::endl;
  output_file_in_0.close();
  std::ofstream output_file_in_1(filepath_0+"intersection_call_in.txt", std::ios::out | std::ios::trunc);
  output_file_in_1 << std::to_string(intersection_call) << std::endl;
  output_file_in_1.close();

  serialize_intersection_data(filepath_0+"contour_intersections_in.json", contour_intersections);
  // 
  // END OF TESTING
  // 

  intersections.clear();
  intersections.resize(image_segments.size());
  intersection_indices.clear();
  intersection_indices.resize(image_segments.size());

  // Setup intersection diagnostic tools
  IntersectionStats intersection_stats;

  // Compute all rational bezier control points
  std::vector<Eigen::Matrix<double, 5, 3>> image_segments_bezier_control_points;
  image_segments_bezier_control_points.reserve(image_segments.size());
  for (size_t i = 0; i < image_segments.size(); i++) {
    Eigen::Matrix<double, 5, 3> bezier_control_points;
    compute_homogeneous_bezier_points_over_interval(
      image_segments[i],
      image_segments[i].domain().get_lower_bound(),
      image_segments[i].domain().get_upper_bound(),
      bezier_control_points);
    image_segments_bezier_control_points.push_back(bezier_control_points);
  }
  // 
  // TESTING
  // 
  std::string filepath_1 = "spot_control/contour_network/intersection_heuristics/compute_homogeneous_bezier_points_over_interval/";
  serialize_vector_rational_function<4, 2>(filepath_1+"image_segments.json", image_segments);
  serialize_vector_matrix_d<5, 3>(filepath_1+"image_segments_bezier_control_points.csv", image_segments_bezier_control_points);
  // 
  // END OF TESTING
  // 


  // Compute all bounding boxes
  std::vector<std::pair<PlanarPoint, PlanarPoint>> image_segments_bounding_box;
  image_segments_bounding_box.reserve(image_segments.size());
  for (size_t i = 0; i < image_segments.size(); i++) {
    PlanarPoint lower_left_point, upper_right_point;
    compute_bezier_bounding_box(
      image_segments[i], lower_left_point, upper_right_point);
    image_segments_bounding_box.push_back(
      std::make_pair(lower_left_point, upper_right_point));
  }
  // 
  // TESTING
  // 
  std::string filepath_2 = "spot_control/contour_network/intersection_heuristics/compute_bezier_bounding_box/";
  serialize_vector_rational_function<4, 2>(filepath_2+"image_segments.json", image_segments);
  serialize_vector_pair_planarpoint(filepath_2+"image_segments_bounding_box.csv", image_segments_bounding_box);
  // 
  // END OF TESTING
  // 


  // Hash by uv
  int num_interval = 50;
  std::vector<int>
    hash_table[50][50]; // FIXME Make global: change both here and num_interval
  std::vector<std::vector<int>> reverse_hash_table;
  compute_bounding_box_hash_table(
    image_segments_bounding_box, hash_table, reverse_hash_table);



  // TESTING
  // static size_t counter = 0;
  

  // Compute intersections
  int num_segments = image_segments.size();
  for (int image_segment_index = 0; image_segment_index < num_segments;
       ++image_segment_index) {
    const std::vector<int>& cells = reverse_hash_table[image_segment_index];
    std::vector<bool> visited(image_segment_index, false);

    for (int cell : cells) {
      int j = cell / num_interval;
      int k = cell % num_interval;
      for (int i : hash_table[j][k]) {
        if (i >= image_segment_index || visited[i])
          continue;
        visited[i] = true;

        // Iterate over image segments with lower indices
        spdlog::trace("Computing segments {}, {} out of {}",
                      image_segment_index,
                      i,
                      image_segments.size());

        // Compute intersections between the two image segments
        // TODO: serialize everything (include the intersection stats... oh great)
        // static size_t counter = 0;
        // std::string filepath = "spot_control/contour_network/compute_intersections/compute_planar_curve_intersections/";
        // serialize_vector_rational_function(
        //   filepath+"first_planar_curve/"+std::to_string(counter)+".json", 
        //   std::vector{image_segments[image_segment_index],});
        // serialize_vector_rational_function(
        //   filepath+"second_planar_curve/"+std::to_string(counter)+".json",
        //   std::vector{image_segments[i]});
        // // NOTE: only need to serialize once
        // serialize_intersection_params(filepath+"intersect_params.json", intersect_params);

        // serialize_intersection_stats(
        //   filepath+"intersection_stats_in/"+std::to_string(counter)+".json",
        //   intersection_stats);
        // serialize_vector_pair_planarpoint(
        //   filepath+"first_bounding_box/"+std::to_string(counter)+".csv",
        //   std::vector{image_segments_bounding_box[image_segment_index]});
        // serialize_vector_pair_planarpoint(
        //   filepath+"second_bounding_box/"+std::to_string(counter)+".csv",
        //   std::vector{image_segments_bounding_box[i]});
        // serialize_eigen_matrix_d(
        //   filepath+"first_bezier_control_points/"+std::to_string(counter)+".csv",
        //   image_segments_bezier_control_points[image_segment_index]);
        // serialize_eigen_matrix_d(
        //   filepath+"second_bezier_control_points/"+std::to_string(counter)+".csv",
        //   image_segments_bezier_control_points[i]);



        std::vector<double> current_segment_intersections;
        std::vector<double> other_segment_intersections;
        compute_planar_curve_intersections(
          image_segments[image_segment_index],
          image_segments[i],
          intersect_params,
          current_segment_intersections,
          other_segment_intersections,
          intersection_stats,
          image_segments_bounding_box[image_segment_index],
          image_segments_bounding_box[i],
          image_segments_bezier_control_points[image_segment_index],
          image_segments_bezier_control_points[i]);

        //   // TESTING: serialize results
        // serialize_vector_int(
        //   filepath+"first_curve_intersections/"+std::to_string(counter)+".csv",
        //   current_segment_intersections);
        // serialize_vector_int(
        //   filepath+"second_curve_intersections/"+std::to_string(counter)+".csv",
        //   other_segment_intersections);
        // serialize_intersection_stats(
        //   filepath+"intersection_stats_out/"+std::to_string(counter)+".json",
        //   intersection_stats);
        //   // END OF TESTING

        append(intersections[image_segment_index],
               current_segment_intersections);
        append(intersections[i], other_segment_intersections);

        // Record the respective indices corresponding to the intersections
        for (size_t k = 0; k < other_segment_intersections.size(); ++k) {
          intersection_indices[image_segment_index].push_back(i);
          intersection_indices[i].push_back(image_segment_index);
        }


        // Build full intersection data
        for (size_t k = 0; k < other_segment_intersections.size(); ++k) {
          IntersectionData current_intersection_data;
          current_intersection_data.knot = current_segment_intersections[k];
          current_intersection_data.intersection_index = i;
          current_intersection_data.intersection_knot =
            other_segment_intersections[k];
          current_intersection_data.id = num_intersections;
          current_intersection_data.check_if_tip(
            image_segments[image_segment_index].domain(),
            intersect_params.trim_amount);
          current_intersection_data.check_if_base(
            image_segments[image_segment_index].domain(),
            intersect_params.trim_amount);
          contour_intersections[image_segment_index].push_back(
            current_intersection_data);

          // Build complementary boundary intersection data
          IntersectionData other_intersection_data;
          other_intersection_data.knot = other_segment_intersections[k];
          other_intersection_data.intersection_index = image_segment_index;
          other_intersection_data.intersection_knot =
            current_segment_intersections[k];
          other_intersection_data.id = num_intersections;
          other_intersection_data.check_if_tip(image_segments[i].domain(),
                                               intersect_params.trim_amount);
          other_intersection_data.check_if_base(image_segments[i].domain(),
                                                intersect_params.trim_amount);
          contour_intersections[i].push_back(other_intersection_data);
          num_intersections++;
        }

        // // TESTING: contour_intersections AFTER building whatnot
        // // TESTING: intersection indices AFTER PUSHBACK...
        // // TESTING: record AFTER APPENDING and whatnot...
        // std::string filepath_3 = "spot_control/contour_network/compute_intersections/compute_intersections/after_compute_curve_intersections/";
        // serialize_vector_vector(filepath_3+"intersections/"+std::to_string(counter)+".csv", intersections);
        // serialize_vector_vector(filepath_3+"intersection_indices/"+std::to_string(counter)+".csv", intersection_indices);
        // serialize_intersection_data(filepath_3+"contour_intersections/"+std::to_string(counter)+".json", contour_intersections);
        // serialize_vector_int(filepath_3+"num_intersections/"+std::to_string(counter)+".csv", std::vector{num_intersections});
        // // Incrementing counter for the next iteration of comparison.
        // // NOTE: need separate counter since we want to track the values for EVERY ITERATION of each inner loop within (and whatnot...)
        // // So, making sure that the behavior for EVERY ITERATION for EVERY NESTED LOOP is the same.
        // counter++;
      }
    }
  }

  // Record intersection information
  intersection_call = intersection_stats.intersection_call;
  SPDLOG_INFO("Number of intersection tests: {}",
              intersection_stats.num_intersection_tests);
  SPDLOG_INFO("Number of nonoverlapping Bezier boxes: {}",
              intersection_stats.num_bezier_nonoverlaps);


  // -----------------
  // TESTING
  // -----------------
  std::string filepath = "spot_control/contour_network/compute_intersections/compute_intersections/";
  serialize_vector_vector(filepath+"intersections.csv", intersections);
  serialize_vector_vector(filepath+"intersection_indices.csv", intersection_indices);

  // SERIALIZING VANILLA JSON 
  serialize_intersection_data(filepath+"contour_intersections_out.json", contour_intersections);
  // std::ofstream output_file_0(filepath+"contour_intersections.json", std::ios::out | std::ios::trunc);
  // output_file_0 << "[\n";

  // // Now for every element..
  // for (size_t i = 0; i < contour_intersections.size(); ++i) {
  //   output_file_0 << "  [\n"; // start inner array
  //   for (size_t j = 0; j < contour_intersections.at(i).size(); ++j) {
  //     output_file_0 << "    {\n";
  //     output_file_0 << "      \"knot\": " << std::setprecision(17) << contour_intersections.at(i).at(j).knot << ",\n";
  //     output_file_0 << "      \"intersection_index\": " + std::to_string(contour_intersections.at(i).at(j).intersection_index) + ",\n";
  //     output_file_0 << "      \"intersection_knot\": " + std::to_string(contour_intersections.at(i).at(j).intersection_knot) + ",\n";
  //     output_file_0 << "      \"is_base\": " + std::to_string(contour_intersections.at(i).at(j).is_base) + ",\n";
  //     output_file_0 << "      \"is_tip\": " + std::to_string(contour_intersections.at(i).at(j).is_tip) + ",\n";
  //     output_file_0 << "      \"id\": " + std::to_string(contour_intersections.at(i).at(j).id) + ",\n";
  //     output_file_0 << "      \"is_redundant\": " << std::boolalpha << contour_intersections.at(i).at(j).is_redundant << "\n";
  //     output_file_0 << "    }";
  //     if (j + 1 != contour_intersections.at(i).size()) output_file_0 << ",";
  //     output_file_0 << "\n";
  //   }
  //   output_file_0 << "  ]";
  //   if (i + 1 != contour_intersections.size()) output_file_0 << ",";
  //   output_file_0 << "\n";
  // }
  // output_file_0 << "]\n";
  // output_file_0.close();

  std::ofstream output_file(filepath+"num_intersections_out.txt", std::ios::out | std::ios::trunc);
  output_file << std::to_string(num_intersections) << std::endl;
  output_file.close();

  std::ofstream output_file_2(filepath+"intersection_call_out.txt", std::ios::out | std::ios::trunc);
  output_file_2 << std::to_string(intersection_call) << std::endl;
  output_file_2.close();
}

void
split_planar_curves_no_self_intersection(
  const std::vector<RationalFunction<4, 2>>& planar_curves,
  std::vector<std::vector<double>>& split_points)
{
  split_points.resize(planar_curves.size());
  for (size_t i = 0; i < planar_curves.size(); i++) {
    if (float_equal_zero(planar_curves[i].domain().get_length(), 1e-6)) {
      spdlog::error("Splitting curve of length {}",
                    planar_curves[i].domain().get_length());
    }
    // Get Bezier points
    Eigen::Matrix<double, 5, 3> bezier_control_points;
    compute_homogeneous_bezier_points_over_interval(
      planar_curves[i],
      planar_curves[i].domain().get_lower_bound(),
      planar_curves[i].domain().get_upper_bound(),
      bezier_control_points);
    Point curve[5];
    for (int i = 0; i < 5; i++) {
      curve[i] = bezier_control_points.row(i);
    }

    // Get splits in the Bezier domain
    std::vector<double> split_points_bezier(0);
    split_bezier_curve_no_self_intersection(curve, 0, 1, split_points_bezier);

    // Get splits in the planar curve domain
    split_points[i].resize(split_points_bezier.size());
    for (size_t j = 0; j < split_points_bezier.size(); j++) {
      split_points[i][j] = convert_spline_to_planar_curve_parameter(
        planar_curves[i], split_points_bezier[j]);
    }
  }
}
