#ifndef WARPING_ERROR
#define WARPING_ERROR

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <set>

class Warping_error {
public:
	Warping_error() {}
	~Warping_error() {}

	bool decide_simple_points(const cv::Mat& patch);

	cv::Mat python2mat_vector_uchar(
		const int height,
		const int width,
		std::vector<int>& data
	);

	cv::Mat extract_patch(
		const cv::Mat& image,
		const std::pair<int, int>& p,
		const int kernel_size
	);

	void update_simple_points(
		const cv::Mat& image,
		const std::pair<int, int>& p,
		const int kernel_size,
		std::set<std::pair<int, int>>& simple_points
	);

	float compute_warping_error(
		const cv::Mat& LSTAR,
		const cv::Mat& T,
		bool display
	);

	int compute_warping_error_python(
		const int height,
		const int width,
		std::vector<int>& data_o,
		std::vector<int>& data_d,
		std::vector<int>& result
	);
};

#endif // !WARPING_ERROR