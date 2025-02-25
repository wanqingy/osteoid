#define PYBIND11_DETAILED_ERROR_MESSAGES

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <optional>
#include <vector>
#include <unordered_map>

namespace py = pybind11;

template <typename LABEL>
py::array decompress_helper(
	const crackle::CrackleHeader& head, 
	const uint8_t* buffer,
	const uint64_t num_bytes,
	int64_t z_start, int64_t z_end,
	size_t parallel,
	const std::optional<uint64_t> label = std::nullopt
) {
	int64_t voxels = head.sx * head.sy;
	z_start = std::max(z_start, static_cast<int64_t>(0));
	if (z_end == -1) {
		z_end = head.sz;
	}
	z_end = std::min(
		std::max(z_end, static_cast<int64_t>(0)), 
		static_cast<int64_t>(head.sz)
	);

	voxels *= z_end - z_start;

	py::array arr;
	if (label.has_value()) {
		arr = py::array_t<uint8_t>(voxels);
		crackle::decompress<LABEL, uint8_t>(
			buffer, num_bytes,
			reinterpret_cast<uint8_t*>(const_cast<void*>(arr.data())),
			z_start, z_end, 
			parallel, label
		);
	}
	else {
		arr = py::array_t<LABEL>(voxels);
		crackle::decompress<LABEL, LABEL>(
			buffer, num_bytes,
			reinterpret_cast<LABEL*>(const_cast<void*>(arr.data())),
			z_start, z_end, 
			parallel, label
		);
	}

	return arr;
}


PYBIND11_MODULE(fastosteoid, m) {
	m.doc() = "Accelerated crackle functions."; 
	m.def("paths", &paths, "Decompress a crackle file into a numpy array.");
}
