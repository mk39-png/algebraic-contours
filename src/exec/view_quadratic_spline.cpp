// Copyright 2023 Adobe Research. All rights reserved.
// To view a copy of the license, visit LICENSE.md.

#include "common.h"
#include "globals.cpp"
#include "apply_transformation.h"
#include "generate_transformation.h"
#include "compute_boundaries.h"
#include "contour_network.h"
#include "twelve_split_spline.h"
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <CLI/CLI.hpp>


int main(int argc, char *argv[])
{
  // Build maps from strings to enums
  std::map<std::string, spdlog::level::level_enum> log_level_map {
    {"trace",    spdlog::level::trace},
    {"debug",    spdlog::level::debug},
    {"info",     spdlog::level::info},
    {"warn",     spdlog::level::warn},
    {"critical", spdlog::level::critical},
    {"off",      spdlog::level::off},
  };

  // To avoid error
  int placeholder = sizeof(*argv) + argc;
  std::cout << placeholder << std::endl;

  // Get command line arguments
  CLI::App app{"Generate smooth occluding contours for a mesh."};
  std::string input_filename = "spot_control_mesh-cleaned_conf_simplified_with_uv.obj";
  std::string output_dir = "./";
  spdlog::level::level_enum log_level = spdlog::level::off;
  Eigen::Matrix<double, 3, 1> color = SKY_BLUE;
  int num_subdivisions = DISCRETIZATION_LEVEL;
  OptimizationParameters optimization_params;
  double weight = optimization_params.position_difference_factor;
  app.add_option("-i,--input", input_filename, "Mesh filepath")
    ->check(CLI::ExistingFile)
    ->required();
  app.add_option("--log_level", log_level, "Level of logging")
    ->transform(CLI::CheckedTransformer(log_level_map, CLI::ignore_case));
  app.add_option("--num_subdivisions", num_subdivisions, "Number of subdivisions")
    ->check(CLI::PositiveNumber);
  app.add_option("-w,--weight", weight, "Fitting weight for the quadratic surface approximation")
    ->check(CLI::PositiveNumber);
  // CLI11_PARSE(app, argc, argv);

  // Set logger level
  spdlog::set_level(log_level);

  // Set optimization parameters
  optimization_params.position_difference_factor = weight;

  // Get input mesh
  Eigen::MatrixXd V, uv, N;
  Eigen::MatrixXi F, FT, FN;
  igl::readOBJ(input_filename, V, uv, N, F, FT, FN);

  // Seeing if rows and cols matches with what I have
  std::cout << "Begin bootleg testing" << std::endl;
  std::cout << V.rows() << " " << V.cols() << std::endl;
  std::cout << uv.rows() << " " << uv.cols() << std::endl;
  std::cout << N.rows() << " " << N.cols() << std::endl;
  std::cout << F.rows() << " " << F.cols() << std::endl;
  std::cout << FT.rows() << " " << FT.cols() << std::endl;
  std::cout << FN.rows() << " " << FN.cols() << std::endl;
  

  // Generate quadratic spline
  spdlog::info("Computing spline surface");
  std::vector<std::vector<int>> face_to_patch_indices;
  std::vector<int> patch_to_face_indices;
  Eigen::SparseMatrix<double> fit_matrix;
  Eigen::SparseMatrix<double> energy_hessian;
  Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>>
      energy_hessian_inverse;
  AffineManifold affine_manifold(F, uv, FT);

  // Now seeing affine manifold and if that's all good
  std::cout << "Affine manifold" << std::endl;
  std::cout << affine_manifold.num_faces() << std::endl;
  std::cout << affine_manifold.num_vertices() << std::endl;
  std::cout << affine_manifold.get_faces().rows() << " " << affine_manifold.get_faces().cols() << std::endl;
  std::cout << affine_manifold.get_halfedge().num_faces() << std::endl;
  std::cout << affine_manifold.get_halfedge().num_edges() << std::endl;
  std::cout << affine_manifold.get_halfedge().num_halfedges() << std::endl;
  std::cout << affine_manifold.get_halfedge().num_vertices() << std::endl;

  TwelveSplitSplineSurface spline_surface(
      V, affine_manifold,
      optimization_params, face_to_patch_indices, patch_to_face_indices,
      fit_matrix, energy_hessian, energy_hessian_inverse);

  // View the mesh
  spline_surface.view(color, num_subdivisions);
  // TODO: check the discretization steps in spline_surface above...
}
