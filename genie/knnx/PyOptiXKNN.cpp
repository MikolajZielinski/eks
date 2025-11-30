#include "CPyOptiXKNN.h"

// *** *** *** *** ***

PYBIND11_MODULE(knnx, m) {
	m.doc() = "OptiX KNN library for Python";

	// *********************************************************************************************

	pybind11::class_<CPyOptiXKNN>(m, "CPyOptiXKNN")
		.def(
			pybind11::init<float>(),
			pybind11::arg("chi_square_squared_radius")
		)
		.def(
			"Fit",
			&CPyOptiXKNN::Fit,
			"Builds the BVH tree for the Gaussian neighbors",
			pybind11::arg("m"),
			pybind11::arg("s"),
			pybind11::arg("q")
		)
		.def(
			"KNeighbors",
			&CPyOptiXKNN::KNeighbors,
			"Determines up to K nearest Gaussian neighbors of the queried points",
			pybind11::arg("queried_points"),
			pybind11::arg("K")
		);
}