#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include <Windows.h>
#include <string.h>

using namespace cv;
using namespace std;

Mat OpenImageDialog()
{
	char name[MAX_PATH] = { 0, };
	OPENFILENAMEA ofn;

	ZeroMemory(&ofn, sizeof(ofn));
	ofn.lStructSize = sizeof(OPENFILENAMEA);
	ofn.hwndOwner = NULL;
	ofn.lpstrFilter = "모든파일 0 * .* 0";
	ofn.lpstrFile = name;
	ofn.nMaxFile = MAX_PATH;
	ofn.Flags = OFN_EXPLORER | OFN_FILEMUSTEXIST | OFN_HIDEREADONLY;
	ofn.lpstrDefExt = "";

	Mat image = Mat();

	if (GetOpenFileNameA(&ofn)) {

		image = imread(name, IMREAD_GRAYSCALE);

		if (image.empty()) {
			cout << "파일 읽기 실패" << endl;
			exit(1);
		}
	}
	else {
		cout << "파일 지정 오류" << endl;
		exit(1);
	}

	return image;
}

void ReadImageToString(Mat img) {

		// 관심 영역
		Rect roi(256, 256, 16, 16);
		// 영상에서 관심영역 추출
		Mat roi_img = img(roi);

		// 영상의 값 출력
		cout << "[roi_img] = " << endl;

		for (int i = 0; i < roi_img.rows; i++) {
			for (int j = 0; j < roi_img.cols; j++) {
				cout.width(4);
				cout << (int)roi_img.at<uchar>(i, j);
			}
			cout << endl;
		}

		// 영상에 사각형 그리기 (관심영역)
		rectangle(img, roi, Scalar(255), 1);
}


// 스케일링
void Scaling(Mat img, Mat& dst, Size size) {

	dst = Mat(size, CV_8U, Scalar(0));
	double ratioY = (double)size.height / img.rows;
	double ratioX = (double)size.width / img.cols;

	// 목적 영상 순회 - 순방향 사상
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			int x = (int)(j * ratioX);
			int y = (int)(i * ratioY);
			dst.at<uchar>(y, x) = img.at<uchar>(i, j);
		}
	}
}

// 1. 최근접 이웃 보간법
void ScalingNearest(Mat img, Mat& dst, Size size) {
	dst = Mat(size, CV_8U, Scalar(0));
	double ratioY = (double)size.height / (img.rows - 1);
	double ratioX = (double)size.width / (img.cols - 1);

	// 목적 영상 순회 - 역방향 사상
	for (int i = 0; i < dst.rows; i++) {
		for (int j = 0; j < dst.cols; j++) {
			//반올림하여 원본 영상에서 가장 근접한 화소를 추출
			int x = (int)cvRound(j / ratioX);
			int y = (int)cvRound(i / ratioY);
			dst.at<uchar>(i, j) = img.at<uchar>(y, x);
		}
	}
}

uchar BilinearValue(Mat img, double x, double y) {

	if (x >= img.cols - 1)
		x--;
	if (y >= img.rows - 1)
		y--;

	// 4개 화소
	Point pt((int)x, (int)y);
	int lt = img.at<uchar>(pt),
		lb = img.at<uchar>(pt + Point(0, 1)),
		rt = img.at<uchar>(pt + Point(1, 0)),
		rb = img.at<uchar>(pt + Point(1, 1));

	// 거리 비율
	double alpha = y - pt.y;	// y축
	double beta = x - pt.x;		// x축

	int M1 = lt + (int)cvRound(alpha * (lb - lt));
	int M2 = rt + (int)cvRound(alpha * (rb - rt));
	int P = M1 + (int)cvRound(beta * (M2 - M1));

	// 실수에서 uchar 형으로 변환
	return saturate_cast<char>(P);
}

// 2. 양선형 보간법
void ScalingBilinear(Mat img, Mat &dst, Size size) {
	dst = Mat(size, img.type(), Scalar(0));
	double ratioY = (double)size.height / img.rows;
	double ratioX = (double)size.width / img.cols;

	// 목적 영상 순회 - 역방향 사상
	for (int i = 0; i < dst.rows; i++) {
		for (int j = 0; j < dst.rows; j++) {
			// 반올림하여 원본영상에서 가장 근접한 화소를 추출
			double x = j / ratioX;
			double y = i / ratioY;

			dst.at<uchar>(i, j) = BilinearValue(img, x, y);
		}
	}
}

int main() {

	auto img = OpenImageDialog();
	Mat scalingImg1, scalingImg2;

	ScalingNearest(img, scalingImg1, Size(1024, 1024));		// 이웃
	ScalingBilinear(img, scalingImg2, Size(1024, 1024));	// 양선형

	// 보간법 적용된 이미지 출력
	imshow("스케일링- 이웃 화소", scalingImg1);
	imshow("스케일링- 양선형", scalingImg2);

	// 화소값 (256,256)에서 16x16 출력
	cout << "최근접 이웃 화소 보간법 16 by 16 화소 출력" << endl;
	ReadImageToString(scalingImg1);

	cout << "\n양선형 보간법 16 by 16 화소 출력" << endl;
	ReadImageToString(scalingImg2);

	waitKey();
}