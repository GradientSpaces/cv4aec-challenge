#include "warping_error.h"

bool Warping_error::decide_simple_points(const cv::Mat& patch) {

	cv::Mat labelImage(patch.size(), CV_32S);
	int comp_foreground = cv::connectedComponents(patch, labelImage, 4) - 1;
	int comp_background = cv::connectedComponents(255 - patch, labelImage, 8) - 1;
	return (comp_foreground == 1) && (comp_background == 1);
}

cv::Mat Warping_error::python2mat_vector_uchar(
	const int height,
	const int width,
	std::vector<int>& data)
{
	const int size = data.size();
	assert(size == height * width);
	cv::Mat image(height, width, CV_32SC1, &data.begin()[0]);
	return image;
}

cv::Mat Warping_error::extract_patch(
	const cv::Mat& image,
	const std::pair<int, int>& p,
	const int kernel_size) 
{
	/*
	@image: input image
	@p: center point of the patch
	@kernel_size: odd integer, the radius will be int(kernel_size/2)
	*/
	int rows = image.rows;
	int cols = image.cols;
	int r = int(kernel_size / 2);
	int x_s = p.first - r;
	int y_s = p.second - r;
	int x_e = std::min(x_s + kernel_size - 1, cols - 1);
	int y_e = std::min(y_s + kernel_size - 1, rows - 1);
	x_s = std::max(x_s, 0);
	y_s = std::max(y_s, 0);
	int width = x_e - x_s + 1;
	int height = y_e - y_s + 1;
	cv::Rect rect(x_s, y_s, width, height);
	return image(rect);
}

void Warping_error::update_simple_points(
	const cv::Mat& image,
	const std::pair<int, int>& p,
	const int kernel_size,
	std::set<std::pair<int, int>>& simple_points)
{
	int rows = image.rows;
	int cols = image.cols;
	int r = int(kernel_size / 2);
	int x_s = p.first - r;
	int y_s = p.second - r;
	int x_e = std::min(x_s + kernel_size - 1, cols - 1);
	int y_e = std::min(y_s + kernel_size - 1, rows - 1);
	x_s = std::max(x_s, 0);
	y_s = std::max(y_s, 0);
	for (int i = x_s; i <= x_e; i++)
		for (int j = y_s; j <= y_e; j++) {
			std::pair<int, int> pt_cur(i, j);
			cv::Mat patch = extract_patch(image, pt_cur, kernel_size);
			if (decide_simple_points(patch)) simple_points.insert(pt_cur);
			else if (simple_points.find(pt_cur) != simple_points.end())
				simple_points.erase(pt_cur);
		}
}

float Warping_error::compute_warping_error(
	const cv::Mat& LSTAR,
	const cv::Mat& T,
	bool display)
{
	cv::Mat img_o = LSTAR.clone();
	cv::Mat img_d = T.clone();
	if (img_o.channels() == 3) cv::cvtColor(img_o, img_o, CV_BGR2GRAY);
	if (img_d.channels() == 3) cv::cvtColor(img_d, img_d, CV_BGR2GRAY);
	const int rows = img_o.rows;
	const int cols = img_o.cols;
	assert(rows == img_d.rows && cols == img_d.cols);

	// ===== Find all difference coordinates
	cv::Mat img_diff, nonZeroCoord, nonZeroCoord_o, nonZeroCoord_d;
	cv::absdiff(img_o, img_d, img_diff);
	cv::findNonZero(img_diff, nonZeroCoord);
	cv::findNonZero(img_o, nonZeroCoord_o);
	cv::findNonZero(img_d, nonZeroCoord_d);
	std::set<std::pair<int, int>> diff_coord;
	for (int i = 0; i < nonZeroCoord.total(); i++)
		diff_coord.insert(std::pair<int, int>(nonZeroCoord.at<cv::Point>(i).x, nonZeroCoord.at<cv::Point>(i).y));
	int normalizer = rows * cols - nonZeroCoord_o.total() + rows * cols - nonZeroCoord_d.total();

	// ===== Find initial simple points
	std::set<std::pair<int, int>> simple_points;
	for (int r = 0; r < rows; ++r) {
		for (int c = 0; c < cols; ++c) {
			cv::Mat p = extract_patch(img_o, std::pair<int, int>(c, r), 3);
			if (decide_simple_points(p)) simple_points.insert(std::pair<int, int>(c, r));
		}
	}

	// ===== Warp ground truth to match the prediction
	bool updated = false;
	cv::Mat img_proc = img_o.clone();
	while (true) {
		if (display) {
			cv::Mat disp = img_proc.clone();
			cv::resize(disp, disp, cv::Size(), 5, 5);
			cv::imshow("Warping error", disp);
			cv::waitKey(30);
		}
		for (auto pt : diff_coord)
			if (simple_points.find(pt) != simple_points.end()) {
				img_proc.at<uchar>(pt.second, pt.first) = img_d.at<uchar>(pt.second, pt.first);
				diff_coord.erase(pt);
				update_simple_points(img_proc, pt, 3, simple_points);
				updated = true;
				break;
			}
		if (!updated) break;
		else updated = false;
	}
	return diff_coord.size() * 1.0 / normalizer;
}

int Warping_error::compute_warping_error_python(
	const int rows,
	const int cols,
	std::vector<int>& data_o,
	std::vector<int>& data_d,
	std::vector<int>& result)
{
	cv::Mat LSTAR = python2mat_vector_uchar(rows, cols, data_o);
	cv::Mat T     = python2mat_vector_uchar(rows, cols, data_d);
	cv::Mat img_o = LSTAR.clone();
	cv::Mat img_d = T.clone();
	img_o.convertTo(img_o, CV_8UC1);
	img_d.convertTo(img_d, CV_8UC1);

	// ===== Find all difference coordinates
	cv::Mat img_diff, nonZeroCoord;
	cv::absdiff(img_o, img_d, img_diff);
	cv::findNonZero(img_diff, nonZeroCoord);
	std::set<std::pair<int, int>> diff_coord;
	for (int i = 0; i < nonZeroCoord.total(); i++)
		diff_coord.insert(std::pair<int, int>(nonZeroCoord.at<cv::Point>(i).x, nonZeroCoord.at<cv::Point>(i).y));

	// ===== Find initial simple points
	std::set<std::pair<int, int>> simple_points;
	for (int r = 0; r < rows; ++r) {
		for (int c = 0; c < cols; ++c) {
			cv::Mat p = extract_patch(img_o, std::pair<int, int>(c, r), 3);
			if (decide_simple_points(p)) simple_points.insert(std::pair<int, int>(c, r));
		}
	}

	// ===== Warp ground truth to match the prediction
	bool updated = false;
	cv::Mat img_proc = img_o.clone();
	while (true) {
		for (auto pt : diff_coord)
			if (simple_points.find(pt) != simple_points.end()) {
				img_proc.at<uchar>(pt.second, pt.first) = img_d.at<uchar>(pt.second, pt.first);
				diff_coord.erase(pt);
				update_simple_points(img_proc, pt, 3, simple_points);
				updated = true;
				break;
			}
		if (!updated) break;
		else updated = false;
	}
	img_proc.convertTo(img_proc, CV_32SC1);
	result = std::vector<int>((int*)img_proc.data, (int*)img_proc.data + rows*cols);
	return diff_coord.size();
}