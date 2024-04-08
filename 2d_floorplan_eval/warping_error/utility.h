#ifndef UTILITY
#define UTILITY

#include <iostream>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <nlohmann/json.hpp>

class Utility {
public:
	Utility() {}
	~Utility() {}

	std::vector<std::vector<std::vector<cv::Point2f>>> read_geometry_OBJ(
		const std::string& path_obj,
		bool cvt2_meters
	);

	std::vector<std::vector<std::vector<cv::Point2f>>> read_geometry_JSON(
		const std::string& path_jsn,
		const std::string& cvt2_units
	);

	std::vector<std::vector<std::vector<cv::Point2f>>> cvt_geometry_format_obj2drw(
		const std::vector<std::vector<std::vector<cv::Point2f>>>& geometry
	);

	void cvt_geometry2list(
		const std::vector<std::vector<std::vector<cv::Point2f>>>& geometry,
		std::vector<std::vector<float>>& x1,
		std::vector<std::vector<float>>& y1,
		std::vector<std::vector<float>>& x2,
		std::vector<std::vector<float>>& y2
	);

	void draw_coord(
		cv::Mat& img,
		std::vector<std::vector<float>>& x1,
		std::vector<std::vector<float>>& y1,
		std::vector<std::vector<float>>& x2,
		std::vector<std::vector<float>>& y2,
		const cv::Scalar color,
		const int thickness,
		const int index
	);

	cv::Mat plot_layers(
		std::vector<std::vector<float>>& x1,
		std::vector<std::vector<float>>& y1,
		std::vector<std::vector<float>>& x2,
		std::vector<std::vector<float>>& y2,
		const int height,
		const int width,
		const int thickness,
		const int index
	);

	void determine_curtain_size_sync(
		std::vector<std::vector<float>>& x1_1,
		std::vector<std::vector<float>>& y1_1,
		std::vector<std::vector<float>>& x2_1,
		std::vector<std::vector<float>>& y2_1,
		std::vector<std::vector<float>>& x1_2,
		std::vector<std::vector<float>>& y1_2,
		std::vector<std::vector<float>>& x2_2,
		std::vector<std::vector<float>>& y2_2,
		int& res_height,
		int& res_width
	);
};

#endif // !UTILITY