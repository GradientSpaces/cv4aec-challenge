#include "warping_error.h"
#include "utility.h"
#include <fstream>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <nlohmann/json.hpp>

int main(int argc, char* argv[]) {

	Utility ut;
	std::string path_jsn1 = argv[1];
	std::string path_jsn2 = argv[2];
	std::vector<std::vector<std::vector<cv::Point2f>>> geometry1 = ut.read_geometry_JSON(path_jsn1, "20cm");
	std::vector<std::vector<std::vector<cv::Point2f>>> geometry2 = ut.read_geometry_JSON(path_jsn2, "20cm");
	std::vector<std::vector<std::vector<cv::Point2f>>> geometry_wrt1 = ut.cvt_geometry_format_obj2drw(geometry1);
	std::vector<std::vector<std::vector<cv::Point2f>>> geometry_wrt2 = ut.cvt_geometry_format_obj2drw(geometry2);
	std::vector<std::vector<float>> x1_1, y1_1, x2_1, y2_1;
	std::vector<std::vector<float>> x1_2, y1_2, x2_2, y2_2;
	ut.cvt_geometry2list(geometry_wrt1, x1_1, y1_1, x2_1, y2_1);
	ut.cvt_geometry2list(geometry_wrt2, x1_2, y1_2, x2_2, y2_2);
	int height, width;
	ut.determine_curtain_size_sync(x1_1, y1_1, x2_1, y2_1, x1_2, y1_2, x2_2, y2_2, height, width);
	cv::Mat img1 = ut.plot_layers(x1_1, y1_1, x2_1, y2_1, height, width, 1, -1);
	cv::Mat img2 = ut.plot_layers(x1_2, y1_2, x2_2, y2_2, height, width, 1, -1);
	cv::flip(img1, img1, 0);
	cv::flip(img2, img2, 0);

	Warping_error we;
	std::cout << we.compute_warping_error(img1, img2, false) << std::endl;

	system("pause");
	return 0;
}