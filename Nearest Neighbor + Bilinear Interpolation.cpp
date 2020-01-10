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
	ofn.lpstrFilter = "������� 0 * .* 0";
	ofn.lpstrFile = name;
	ofn.nMaxFile = MAX_PATH;
	ofn.Flags = OFN_EXPLORER | OFN_FILEMUSTEXIST | OFN_HIDEREADONLY;
	ofn.lpstrDefExt = "";

	Mat image = Mat();

	if (GetOpenFileNameA(&ofn)) {

		image = imread(name, IMREAD_GRAYSCALE);

		if (image.empty()) {
			cout << "���� �б� ����" << endl;
			exit(1);
		}
	}
	else {
		cout << "���� ���� ����" << endl;
		exit(1);
	}

	return image;
}

void ReadImageToString(Mat img) {

		// ���� ����
		Rect roi(256, 256, 16, 16);
		// ���󿡼� ���ɿ��� ����
		Mat roi_img = img(roi);

		// ������ �� ���
		cout << "[roi_img] = " << endl;

		for (int i = 0; i < roi_img.rows; i++) {
			for (int j = 0; j < roi_img.cols; j++) {
				cout.width(4);
				cout << (int)roi_img.at<uchar>(i, j);
			}
			cout << endl;
		}

		// ���� �簢�� �׸��� (���ɿ���)
		rectangle(img, roi, Scalar(255), 1);
}


// �����ϸ�
void Scaling(Mat img, Mat& dst, Size size) {

	dst = Mat(size, CV_8U, Scalar(0));
	double ratioY = (double)size.height / img.rows;
	double ratioX = (double)size.width / img.cols;

	// ���� ���� ��ȸ - ������ ���
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			int x = (int)(j * ratioX);
			int y = (int)(i * ratioY);
			dst.at<uchar>(y, x) = img.at<uchar>(i, j);
		}
	}
}

// 1. �ֱ��� �̿� ������
void ScalingNearest(Mat img, Mat& dst, Size size) {
	dst = Mat(size, CV_8U, Scalar(0));
	double ratioY = (double)size.height / (img.rows - 1);
	double ratioX = (double)size.width / (img.cols - 1);

	// ���� ���� ��ȸ - ������ ���
	for (int i = 0; i < dst.rows; i++) {
		for (int j = 0; j < dst.cols; j++) {
			//�ݿø��Ͽ� ���� ���󿡼� ���� ������ ȭ�Ҹ� ����
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

	// 4�� ȭ��
	Point pt((int)x, (int)y);
	int lt = img.at<uchar>(pt),
		lb = img.at<uchar>(pt + Point(0, 1)),
		rt = img.at<uchar>(pt + Point(1, 0)),
		rb = img.at<uchar>(pt + Point(1, 1));

	// �Ÿ� ����
	double alpha = y - pt.y;	// y��
	double beta = x - pt.x;		// x��

	int M1 = lt + (int)cvRound(alpha * (lb - lt));
	int M2 = rt + (int)cvRound(alpha * (rb - rt));
	int P = M1 + (int)cvRound(beta * (M2 - M1));

	// �Ǽ����� uchar ������ ��ȯ
	return saturate_cast<char>(P);
}

// 2. �缱�� ������
void ScalingBilinear(Mat img, Mat &dst, Size size) {
	dst = Mat(size, img.type(), Scalar(0));
	double ratioY = (double)size.height / img.rows;
	double ratioX = (double)size.width / img.cols;

	// ���� ���� ��ȸ - ������ ���
	for (int i = 0; i < dst.rows; i++) {
		for (int j = 0; j < dst.rows; j++) {
			// �ݿø��Ͽ� �������󿡼� ���� ������ ȭ�Ҹ� ����
			double x = j / ratioX;
			double y = i / ratioY;

			dst.at<uchar>(i, j) = BilinearValue(img, x, y);
		}
	}
}

int main() {

	auto img = OpenImageDialog();
	Mat scalingImg1, scalingImg2;

	ScalingNearest(img, scalingImg1, Size(1024, 1024));		// �̿�
	ScalingBilinear(img, scalingImg2, Size(1024, 1024));	// �缱��

	// ������ ����� �̹��� ���
	imshow("�����ϸ�- �̿� ȭ��", scalingImg1);
	imshow("�����ϸ�- �缱��", scalingImg2);

	// ȭ�Ұ� (256,256)���� 16x16 ���
	cout << "�ֱ��� �̿� ȭ�� ������ 16 by 16 ȭ�� ���" << endl;
	ReadImageToString(scalingImg1);

	cout << "\n�缱�� ������ 16 by 16 ȭ�� ���" << endl;
	ReadImageToString(scalingImg2);

	waitKey();
}