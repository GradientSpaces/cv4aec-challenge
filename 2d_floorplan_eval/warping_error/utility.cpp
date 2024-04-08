#include "utility.h"

std::vector<std::vector<std::vector<cv::Point2f>>> Utility::read_geometry_OBJ(const std::string& path_obj, bool cvt2_meters) {
	std::ifstream in(path_obj.c_str(), std::ios::binary | std::ios::in);
	if (!in.good()) { std::cout << "File open failed ..." << std::endl; exit(1); }

	int num_layers;
	in.read(reinterpret_cast<char*>(&num_layers), 4);
	std::vector<int> num_struct(num_layers);
	in.read(reinterpret_cast<char*>(num_struct.data()), 4 * num_layers);
	std::vector<std::vector<std::vector<cv::Point2f>>> geometry(num_layers);
	for (int i = 0; i < num_layers; i++)
		geometry[i] = std::vector<std::vector<cv::Point2f>>(num_struct[i]);
	for (int i = 0; i < num_layers; i++) {
		int layer_name_length;
		in.read(reinterpret_cast<char*>(&layer_name_length), 4);
		char* layer_name_ = new char(layer_name_length);
		in.read(reinterpret_cast<char*>(layer_name_), layer_name_length);
		std::string layer_name(layer_name_);
		for (int j = 0; j < num_struct[i]; j++) {
			int pt_num;
			in.read(reinterpret_cast<char*>(&pt_num), 4);
			geometry[i][j].reserve(pt_num);
			for (int k = 0; k < pt_num; k++) {
				float x_, y_;
				in.read(reinterpret_cast<char*>(&x_), 4);
				in.read(reinterpret_cast<char*>(&y_), 4);
				if (cvt2_meters) { x_ = x_ / 0.3048 * 12; y_ = y_ / 0.3048 * 12; }
				geometry[i][j].push_back(cv::Point2f(x_, y_));
			}
		}
		delete layer_name_;
	}
	num_struct.clear();
	in.clear();
	in.close();
	return geometry;
}

std::vector<std::vector<std::vector<cv::Point2f>>> Utility::read_geometry_JSON(const std::string& path_jsn, const std::string& cvt2_units) {
	boost::property_tree::ptree root;
	boost::property_tree::read_json(path_jsn.c_str(), root);
	// Read in header
	int num_layers;
	std::vector<int> num_struct;
	for (auto& node : root.get_child("header")) {
		if (node.first == "layer number") num_layers = (int)node.second.get_value<int>();
		else {
			num_struct.reserve(num_layers);
			const boost::property_tree::ptree& subtree = node.second;
			for (auto& subnode : subtree)
				num_struct.push_back(subnode.second.get_value<int>());
		}
	}

	std::vector<std::vector<std::vector<cv::Point2f>>> geometry(num_layers);
	for (int i = 0; i < num_layers; i++)
		geometry[i] = std::vector<std::vector<cv::Point2f>>(num_struct[i]);
	for (int i = 0; i < num_layers; i++) {
		int struct_idx = 0;
		std::string layer_index = "layer " + std::to_string(i);
		for (auto& node : root.get_child(layer_index.c_str())) {
			std::string layer_name;
			if (node.first == "layer name") layer_name = node.second.get_value<std::string>();
			else {
				const boost::property_tree::ptree& subtree = node.second;
				for (auto& subnode : subtree) {
					const boost::property_tree::ptree& subsubtree = subnode.second;
					for (auto& subsubnode : subsubtree) {
						int pt_num;
						if (subsubnode.first == "point number") pt_num = subsubnode.second.get_value<int>();
						else {
							const boost::property_tree::ptree& subsubsubtree = subsubnode.second;
							geometry[i][struct_idx].reserve(pt_num);
							std::vector<float> temp_;
							temp_.reserve(2 * pt_num);
							for (auto& subsubsubnode : subsubsubtree)
								temp_.push_back(subsubsubnode.second.get_value<float>());
							for (int j = 0; j < pt_num; j++)
								if (cvt2_units.compare(std::string("1inch")) == 0)
									geometry[i][struct_idx].push_back(cv::Point2f(temp_[j * 2] / 0.3048 * 12, temp_[j * 2 + 1] / 0.3048 * 12));
								else if (cvt2_units.compare(std::string("1cm")) == 0)
									geometry[i][struct_idx].push_back(cv::Point2f(temp_[j * 2] * 100, temp_[j * 2 + 1] * 100));
								else if (cvt2_units.compare(std::string("10cm")) == 0)
									geometry[i][struct_idx].push_back(cv::Point2f(temp_[j * 2] * 10, temp_[j * 2 + 1] * 10));
								else if (cvt2_units.compare(std::string("20cm")) == 0)
									geometry[i][struct_idx].push_back(cv::Point2f(temp_[j * 2] * 5, temp_[j * 2 + 1] * 5));
								else if (cvt2_units.compare(std::string("1m")) == 0)
									geometry[i][struct_idx].push_back(cv::Point2f(temp_[j * 2], temp_[j * 2 + 1]));
								else {
									std::cout << "Invalid units request ..." << std::endl;
									exit(1);
								}
							struct_idx++;
						}
					}
				}
			}
		}
	}
	return geometry;
}

std::vector<std::vector<std::vector<cv::Point2f>>> Utility::cvt_geometry_format_obj2drw(const std::vector<std::vector<std::vector<cv::Point2f>>>& geometry) {
	const int layer_num = geometry.size();
	std::vector<std::vector<std::vector<cv::Point2f>>> res(layer_num);
	for (int i = 0; i < layer_num; i++) {
		const int struct_num = geometry[i].size();
		for (int j = 0; j < struct_num; j++) {
			const int point_num = geometry[i][j].size();
			for (int k = 0; k < point_num - 1; k++) {
				std::vector<cv::Point2f> tmp_ = { geometry[i][j][k], geometry[i][j][k + 1] };
				res[i].push_back(tmp_);
			}
		}
	}
	return res;
}

void Utility::cvt_geometry2list(
	const std::vector<std::vector<std::vector<cv::Point2f>>>& geometry,
	std::vector<std::vector<float>>& x1,
	std::vector<std::vector<float>>& y1,
	std::vector<std::vector<float>>& x2,
	std::vector<std::vector<float>>& y2)
{
	const int layer_num = geometry.size();
	x1.resize(layer_num);
	y1.resize(layer_num);
	x2.resize(layer_num);
	y2.resize(layer_num);
	for (int i = 0; i < layer_num; i++) {
		const int t = geometry[i].size();
		x1[i].reserve(t);
		y1[i].reserve(t);
		x2[i].reserve(t);
		y2[i].reserve(t);
		for (int j = 0; j < t; j++) {
			x1[i].push_back(geometry[i][j][0].x);
			y1[i].push_back(geometry[i][j][0].y);
			x2[i].push_back(geometry[i][j][1].x);
			y2[i].push_back(geometry[i][j][1].y);
		}
	}
}

void Utility::draw_coord(
	cv::Mat& img,
	std::vector<std::vector<float>>& x1,
	std::vector<std::vector<float>>& y1,
	std::vector<std::vector<float>>& x2,
	std::vector<std::vector<float>>& y2,
	const cv::Scalar color,
	const int thickness,
	const int index
)
{
	assert(index < x1.size());
	const int rows = img.rows;
	const int cols = img.cols;
	for (int i = 0; i < x1[index].size(); i++) {
		int x1_ = int(std::rint(x1[index][i]));
		int y1_ = int(std::rint(y1[index][i]));
		int x2_ = int(std::rint(x2[index][i]));
		int y2_ = int(std::rint(y2[index][i]));
		if (x1_ < 0 || x1_ >= cols || x2_ < 0 || x2_ >= cols || y1_ < 0 || y1_ >= rows || y2_ < 0 || y2_ >= rows)
			std::cout << "Invalid coordinates" << std::endl;
		cv::line(img, cv::Point(x1_, y1_), cv::Point(x2_, y2_), color, thickness, cv::LINE_8);
	}
}

cv::Mat Utility::plot_layers(
	std::vector<std::vector<float>>& x1,
	std::vector<std::vector<float>>& y1,
	std::vector<std::vector<float>>& x2,
	std::vector<std::vector<float>>& y2,
	const int height,
	const int width,
	const int thickness,
	const int index
)
{
	/*index: indicator of which layer to draw, draw all layers if -1
	*/
	cv::Mat img(height, width, CV_8UC3, cv::Scalar(255, 255, 255));
	const int layer_num = x1.size();
	if (layer_num == 0) return img;
	assert(index < layer_num);
	if (index == -1) {
		std::vector<cv::Scalar> colors(layer_num);
		colors[0] = cv::Scalar(0, 0, 0);
		for (int i = 1; i < layer_num; i++)
			//colors[i] = cv::Scalar(std::rand() % 256, std::rand() % 256, std::rand() % 256);
			colors[i] = cv::Scalar(0, 0, 0);
		for (int i = 0; i < layer_num; i++)
			draw_coord(img, x1, y1, x2, y2, colors[i], thickness, i);
	}
	else {
		cv::Scalar color = cv::Scalar(0, 0, 0);
		draw_coord(img, x1, y1, x2, y2, color, thickness, index);
	}
	return img;
}

void Utility::determine_curtain_size_sync(
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
)
{
	/*index: indicator of which layer to draw, draw all layers if -1
	*/
	const int layer_num_1 = x1_1.size();
	const int layer_num_2 = x1_2.size();

	// ===== Decide curtain size =====
	std::vector<float> x_1_min(layer_num_1 * 2, std::numeric_limits<float>::max());
	std::vector<float> y_1_min(layer_num_1 * 2, std::numeric_limits<float>::max());
	std::vector<float> x_1_max(layer_num_1 * 2, std::numeric_limits<float>::min());
	std::vector<float> y_1_max(layer_num_1 * 2, std::numeric_limits<float>::min());
	std::vector<float> x_2_min(layer_num_2 * 2, std::numeric_limits<float>::max());
	std::vector<float> y_2_min(layer_num_2 * 2, std::numeric_limits<float>::max());
	std::vector<float> x_2_max(layer_num_2 * 2, std::numeric_limits<float>::min());
	std::vector<float> y_2_max(layer_num_2 * 2, std::numeric_limits<float>::min());

	for (int i = 0; i < layer_num_1; i++) {
		if (x1_1[i].size() > 0) {
			x_1_min[i * 2 + 0] = *std::min_element(x1_1[i].begin(), x1_1[i].end());
			x_1_min[i * 2 + 1] = *std::min_element(x2_1[i].begin(), x2_1[i].end());
			y_1_min[i * 2 + 0] = *std::min_element(y1_1[i].begin(), y1_1[i].end());
			y_1_min[i * 2 + 1] = *std::min_element(y2_1[i].begin(), y2_1[i].end());
		}
	}
	for (int i = 0; i < layer_num_2; i++) {
		if (x1_2[i].size() > 0) {
			x_2_min[i * 2 + 0] = *std::min_element(x1_2[i].begin(), x1_2[i].end());
			x_2_min[i * 2 + 1] = *std::min_element(x2_2[i].begin(), x2_2[i].end());
			y_2_min[i * 2 + 0] = *std::min_element(y1_2[i].begin(), y1_2[i].end());
			y_2_min[i * 2 + 1] = *std::min_element(y2_2[i].begin(), y2_2[i].end());
		}
	}

	float x_1_min_v = std::numeric_limits<float>::max(), y_1_min_v = std::numeric_limits<float>::max();
	float x_2_min_v = std::numeric_limits<float>::max(), y_2_min_v = std::numeric_limits<float>::max();
	if (layer_num_1 > 0) { 
		x_1_min_v = *std::min_element(x_1_min.begin(), x_1_min.end());
		y_1_min_v = *std::min_element(y_1_min.begin(), y_1_min.end());
	}
	if (layer_num_2 > 0) {
		x_2_min_v = *std::min_element(x_2_min.begin(), x_2_min.end());
		y_2_min_v = *std::min_element(y_2_min.begin(), y_2_min.end());
	}
	float x_shift = 50.0 - std::min(x_1_min_v, x_2_min_v);
	float y_shift = 50.0 - std::min(y_1_min_v, y_2_min_v);

	for (int i = 0; i < layer_num_1; i++) {
		const int t = x1_1[i].size();
		for (int j = 0; j < t; j++) {
			x1_1[i][j] += x_shift;
			x2_1[i][j] += x_shift;
			y1_1[i][j] += y_shift;
			y2_1[i][j] += y_shift;
		}
	}
	for (int i = 0; i < layer_num_2; i++) {
		const int t = x1_2[i].size();
		for (int j = 0; j < t; j++) {
			x1_2[i][j] += x_shift;
			x2_2[i][j] += x_shift;
			y1_2[i][j] += y_shift;
			y2_2[i][j] += y_shift;
		}
	}

	for (int i = 0; i < layer_num_1; i++) {
		if (x1_1[i].size() > 0) {
			x_1_max[i * 2 + 0] = *std::max_element(x1_1[i].begin(), x1_1[i].end());
			x_1_max[i * 2 + 1] = *std::max_element(x2_1[i].begin(), x2_1[i].end());
			y_1_max[i * 2 + 0] = *std::max_element(y1_1[i].begin(), y1_1[i].end());
			y_1_max[i * 2 + 1] = *std::max_element(y2_1[i].begin(), y2_1[i].end());
		}
	}
	for (int i = 0; i < layer_num_2; i++) {
		if (x1_2[i].size() > 0) {
			x_2_max[i * 2 + 0] = *std::max_element(x1_2[i].begin(), x1_2[i].end());
			x_2_max[i * 2 + 1] = *std::max_element(x2_2[i].begin(), x2_2[i].end());
			y_2_max[i * 2 + 0] = *std::max_element(y1_2[i].begin(), y1_2[i].end());
			y_2_max[i * 2 + 1] = *std::max_element(y2_2[i].begin(), y2_2[i].end());
		}
	}

	float x_1_max_v = std::numeric_limits<float>::min(), y_1_max_v = std::numeric_limits<float>::min();
	float x_2_max_v = std::numeric_limits<float>::min(), y_2_max_v = std::numeric_limits<float>::min();
	if (layer_num_1 > 0) {
		x_1_max_v = *std::max_element(x_1_max.begin(), x_1_max.end());
		y_1_max_v = *std::max_element(y_1_max.begin(), y_1_max.end());
	}
	if (layer_num_2 > 0) {
		x_2_max_v = *std::max_element(x_2_max.begin(), x_2_max.end());
		y_2_max_v = *std::max_element(y_2_max.begin(), y_2_max.end());
	}
	res_width  = int(std::ceil(std::max(x_1_max_v, x_2_max_v))) + 50;
	res_height = int(std::ceil(std::max(y_1_max_v, y_2_max_v))) + 50;
}