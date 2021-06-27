
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <stdio.h>
#include<math.h>

using namespace cv;
using namespace std;

int main(int ac, char** av) {

	Mat f= imread("homo1.jpg",0); //두번째 파라미터로 0주면 흑백으로 read 
	imshow("image", f);
	f.convertTo(f, CV_32F, 1/255.f);
	f += 0.1;
	log(f, f);
	Mat F;
	dft(f, F, DFT_COMPLEX_OUTPUT);
	//log(img + 0.1);
	Mat filter = Mat::zeros(F.size(), CV_32FC2);
	for (int y = 0;y < filter.rows;y++) for (int x = 0; x < filter.cols;x++) {
		int xx = x > filter.cols / 2 ? x - filter.cols : x;
		int yy = y > filter.rows / 2 ? y - filter.rows : y;

		float Duv = sqrtf(xx * xx + yy * yy);
		float D0 = 3;

		float Huv = (1.0 - 0.4) * (1 - pow(2.71828, -1 * (Duv * Duv / (D0 * D0)))) + 0.4;

		filter.at<Vec2f>(y, x)[0] = Huv;//*0.8 + 1;
		filter.at<Vec2f>(y, x)[1] = Huv;//*0.8 + 1;
		
	}
	std::vector<Mat> channels;
	multiply(F, filter, F);
	Mat g;
	idft(F, g, DFT_SCALE | DFT_REAL_OUTPUT);
	exp(g, g);
	g -= 0.1;

	split(g, channels);
	imshow("res", channels[0]);
	waitKey(0);
	
	return 0;

}

