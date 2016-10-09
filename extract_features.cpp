#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdlib.h>
#include <string>

#include <cv.h>
#include <highgui.h>
#include "opencv2/features2d/features2d.hpp"

#include "training_data.h"

using namespace cv;
using namespace std;

vector<string> get_training_data()
{
	vector<string> training_data;
	for(int i=0; i<positive_train_size; i++)
	{
		training_data.push_back(positive_train[i]);
	}
	for(int i=0; i<negative_train_size; i++)
	{
		training_data.push_back(negative_train[i]);
	}
	return training_data;
}

vector<string> get_test_data()
{
	vector<string> test_data;
	for(int i=0; i<positive_test_size; i++)
	{
		test_data.push_back(positive_test[i]);
	}
	for(int i=0; i<negative_test_size; i++)
	{
		test_data.push_back(negative_test[i]);
	}
	return test_data;
}

void findLBP(Mat& original, Mat& lbp)
{
	lbp = Mat::zeros(original.rows, original.cols, CV_8UC1);
	int lbp_kernel_size = 3;
	int cmp_count = lbp_kernel_size*lbp_kernel_size - 1;
	for(int i=0; i<original.rows; i++)
	{
		uchar* row_ptr = original.ptr<uchar>(i);
		uchar* lbp_row_ptr = lbp.ptr<uchar>(i);
		for(int j=0; j<original.cols; j++)
		{
			int loc[] = {-1,-1};
			int cur_idx = 0;
			bool switched = false;
			int direct = 1;
			int lbp_val = 1;
			for (int n=0; n<cmp_count; n++)
			{
				switched = (loc[cur_idx] == 0);
				loc[cur_idx] += direct;
				if(switched)
				{
					cur_idx ^= 1;
					if(loc[cur_idx] < 0) direct = 1;
					else direct = -1;
				}
				int locx = i + loc[0];
				int locy = j + loc[1];
				int cmp = 0;
				if (locx < 0 || locx > original.rows || locy < 0 || locy > original.cols)
				{
					cmp = 0;
				}
				else
				{
					cmp = row_ptr[j] < original.ptr<uchar>(locx)[locy] ? 1 : 0;
				}
				lbp_val |= (int)(cmp * pow(2,n));
			}
			lbp_row_ptr[j] = lbp_val;
		}
	}
}

int find_min(vector<int> vec, int val)
{
	// cout << "find: " << val << endl;
	// cout << "size: " << vec.size() << endl;
	for(int i=0; i<vec.size(); i++)
	{
		// cout << vec[i] << endl;
		if(vec[i] >= val)
		{
			return i;
		}
		// else if(vec[i] > val)
		// {
		// 	return i-1;
		// }
	}
	return -1;
}

float euclidean_distance(const Mat& m1, const Mat& m2)
{
	float sum = 0.f;
	for(int i=0; i<m1.cols; i++)
	{
		float diff = m1.at<float>(0,i) - m2.at<float>(0,i);
		sum += diff*diff;
	}
	return sqrt(sum);
}

int find_closest_center(vector<Mat> centers, const Mat& sample)
{
	float min_dist = std::numeric_limits<double>::infinity();
	int min_k = -1;
	for(int k=0; k<centers.size(); k++)
	{
		float dist = euclidean_distance(sample, centers[k]);
		if(dist<min_dist)
		{
			min_dist = dist;
			min_k = k;
		}
	}
	return min_k;
}

vector<Mat> kmeans(int k_clusters, Mat& orig_descriptors, int* cluster_map)
{
	Mat descriptors;
	orig_descriptors.convertTo(descriptors, CV_32FC1);
	int dim = descriptors.cols;
	int samples = descriptors.rows;
	cout << dim << endl;
	cout << samples << endl;

	cout << descriptors.row(0) << endl;
	// cout << descriptors.col(0) << endl;

	srand(0);
	vector<Mat> centers;
	for(int i=0; i<k_clusters; i++)
	{
		int idx = rand() % samples;
		Mat desc_copy = descriptors.row(idx);//.clone()
		Mat center;
		desc_copy.convertTo(center, CV_32FC1);
		centers.push_back(center);
	}

	bool converged = false;
	int iter = 0;
	int max_iter = 10000;
	float epsilon = 0.0001f;
	// int cluster_map[samples];
	Mat cluster_means = Mat::zeros(k_clusters, dim, CV_32FC1);
	int cluster_means_counts[k_clusters];
	while (!converged and iter<max_iter)
	{
		// find closest center for each sample
		for(int i=0; i<samples; i++)
		{
			cluster_map[i] = find_closest_center(centers, descriptors.row(i));
		}

		Mat cluster_means = Mat::zeros(k_clusters, dim, CV_32FC1);
		int cluster_means_counts[k_clusters];
		for(int i=0; i<k_clusters; i++)
		{
			cluster_means_counts[i] = 0;
		}

		// move centers to mean
		for(int i=0; i<samples; i++)
		{
			float* row_ptr = cluster_means.ptr<float>(cluster_map[i]);
			float* desc_ptr = descriptors.ptr<float>(i);
			for(int j=0; j<dim; j++)
			{
				row_ptr[j] += desc_ptr[j];
				// row_ptr[j] += descriptors.at(i, j)
			}
			cluster_means_counts[cluster_map[i]]++;
			// cluster_means.ptr<float>(i)[j] += descriptors.row(idx);
		}

		converged = true;
		for(int i=0; i<k_clusters; i++)
		{
			float delta = 0.f;
			float* mean_ptr = cluster_means.ptr<float>(i);
			float* center_ptr = centers[i].ptr<float>(0);
			// float* center_ptr = centers.ptr<float>(i);
			for(int j=0; j<dim; j++)
			{
				float new_center = mean_ptr[j] / cluster_means_counts[i];
				float diff = center_ptr[j]-new_center;
				delta += diff*diff;
				center_ptr[j] = new_center;
			}
			if(sqrt(delta)>epsilon) converged = false;
		}
		iter++;
		cout << iter << endl;
	}
	return centers;
}

void write_orb_training_features()
{
	vector<string> training_data = get_training_data();
	int training_data_size = training_data.size();

	Mat total_orb_desc;
	vector<KeyPoint> total_orb_keypoints;
	vector<int> keypoint_counts;
	for(int sample_idx=0; sample_idx<training_data_size; sample_idx++)
	{
		Mat img;
		img = imread(training_data[sample_idx], CV_LOAD_IMAGE_GRAYSCALE);
		// Oriented FAST and Rotated BRIEF (ORB) Interest Points
		OrbFeatureDetector orb;
		vector<KeyPoint> orb_keypoints;
		orb.detect(img, orb_keypoints);

		total_orb_keypoints.insert(total_orb_keypoints.end(), orb_keypoints.begin(), orb_keypoints.end());
		int size_integral = 0;
		if(sample_idx!=0) size_integral = keypoint_counts[sample_idx-1];
		keypoint_counts.push_back(size_integral + orb_keypoints.size());
		Mat orb_desc;
		orb.compute(img, orb_keypoints, orb_desc);
		
		if(sample_idx>0)
		{
			if(total_orb_desc.cols != orb_desc.cols) cerr << "MISMATCHED DESCRIPTOR SIZE" << total_orb_desc.cols << "," << orb_desc.cols <<endl;
			if(orb_desc.rows<=2) cerr << "WE NEED TO THROW OUT THIS DATA: " << training_data[sample_idx] << endl;
			Mat tmp_total;
			vconcat(total_orb_desc, orb_desc, tmp_total);
			total_orb_desc = tmp_total;
		}
		else
		{
			total_orb_desc = orb_desc;
		}
		cout << "rows" << endl;
		cout << total_orb_desc.rows << endl;
		cout << "cols" << endl;
		cout << total_orb_desc.cols << endl;
	}

	// cout << orb_keypoints.size() << endl;
	// cout << orb_desc.size() << endl;
	int cluster_map[total_orb_desc.rows];
	int k_clusters = 500;
	vector<Mat> centers = kmeans(k_clusters, total_orb_desc, cluster_map);
	for(int i=0; i<centers.size(); i++)
	{
		cout << centers[i] << endl;
	}
	int example_count = 10;
	int examples[] = {9, 19, 25, 33, 81, 114, 243, 324, 401, 488};

	int word_count_max = 20;
	int word_count[example_count];
	for(int w=0; w<example_count; w++)
		word_count[w] = 0;

	for(int eg=0; eg<example_count; eg++)
	{
	for(int i=0; i<total_orb_desc.rows; i++)
	{
		if(cluster_map[i]==examples[eg])
		{
			int img_idx = find_min(keypoint_counts, i);
			cout << img_idx << endl;
			cout << training_data[img_idx] << endl;
			Mat img = imread(training_data[img_idx], CV_LOAD_IMAGE_GRAYSCALE);
			KeyPoint kp = total_orb_keypoints[i];
			cout << "kp.size: " << kp.size << endl;
			int ix = kp.pt.x - kp.size/2;
			ix = ix<0 ? 0 : ix;
			int iy = kp.pt.y - kp.size/2;
			iy = iy<0 ? 0 : iy;
			if (ix >= img.rows || iy >= img.cols || ix+kp.size/2>=img.rows || iy+kp.size/2>=img.cols) continue;
			cout << "(" << img.rows <<", " << img.cols << ")" << endl;
			int sizex = ix+kp.size/2>=img.rows ? (img.rows - ix - 1)*2: kp.size;
			int sizey = iy+kp.size/2>=img.cols ? (img.cols - iy - 1)*2: kp.size;
			cerr << ix << ", " << iy << ", " << sizex << ", " << sizey << endl;
			imwrite("images/orb_cluster_"+std::to_string(eg)+"_"+std::to_string(word_count[eg])+".ppm", img(Rect(ix, iy, sizex, sizey)));
			cerr << "done" << endl;
			word_count[eg]++;
		}
		if (word_count[eg] >= word_count_max)
		{
			break;
		}
	}
	}

	float** orb_features = new float*[training_data_size];
	for(int i=0; i<training_data_size; i++)
	{
		orb_features[i] = new float[k_clusters];
		for(int j=0; j<k_clusters; j++)
		{
			orb_features[i][j] = 0.f;
		}
	}
	int ipt_idx_start = 0;
	for(int i=0; i<training_data_size; i++)
	{
		int ipt_idx_end = keypoint_counts[i];

		for(int j=ipt_idx_start; j<ipt_idx_end; j++)
		{
			orb_features[i][cluster_map[j]]++;
		}
		// for(int j=ipt_idx_start; j<ipt_idx_end; j++)
		// {
		// 	orb_features[i][cluster_map[j]] /= (ipt_idx_end - ipt_idx_start);
		// }
		ipt_idx_start = ipt_idx_end;
	}

	ofstream feature_file;
	feature_file.open("orb_training_features_large.txt");
	for(int i=0; i<training_data_size; i++)
	{
		for(int j=0; j<k_clusters; j++)
		{
			feature_file << orb_features[i][j] << " ";
		}
		feature_file << endl;
	}
	feature_file.close();

	ofstream center_file;
	center_file.open("orb_kmeans_centers_large.txt");
	for(int i=0; i<centers.size(); i++)
	{
		for(int j=0; j<centers[i].cols; j++)
		{
			center_file << centers[i].at<float>(0,j) << " ";
		}
		center_file << endl;
	}
}

void write_orb_test_features()
{
	string line;
	ifstream centers_file("orb_kmeans_centers_large.txt");
	vector<Mat> centers;
	cout << "open file" << endl;
	while(getline(centers_file, line))
	{
		// cout << "line: " << line << endl;
		string token;
		stringstream ss;
		ss.str(line);
		vector<float> point;
		while(ss.peek() != -1)
		{
			getline(ss, token, ' ');
			point.push_back(stof(token));
		}
		// cout << "length: " << point.size() << endl;
		Mat center;
		center = Mat::zeros(1, point.size(), CV_32FC1);
		// cout << "allocated" << endl;
		for(int i=0; i<point.size(); i++)
		{
			// cout << "accessing point" << endl;
			// cout << point[i] << endl;
			center.ptr<float>(0)[i] = point[i];
			// cout << "assigned" << endl;
		}
		// cout << "push_back" << endl;
		centers.push_back(center);
	}
	cout << centers.size() << endl;
	centers_file.close();

	vector<string> test_data = get_test_data();
	int test_data_size = test_data.size();

	int k_clusters = centers.size();
	int** orb_features = new int*[test_data_size];
	for(int x=0; x<test_data_size; x++)
	{
		orb_features[x] = new int[k_clusters];
		for(int y=0; y<k_clusters; y++)
		{
			orb_features[x][y] = 0;
		}
	}

	for(int i=0; i<test_data_size; i++)
	{
		Mat img;
		img = imread(test_data[i], CV_LOAD_IMAGE_GRAYSCALE);
		OrbFeatureDetector orb;
		vector<KeyPoint> orb_keypoints;
		orb.detect(img, orb_keypoints);
		Mat orb_desc;
		orb.compute(img, orb_keypoints, orb_desc);

		for(int j=0; j<orb_desc.rows; j++)
		{
			Mat desc_copy = orb_desc.row(j);//.clone()
			Mat sample;
			desc_copy.convertTo(sample, CV_32FC1);

			int cluster_idx = find_closest_center(centers, sample);
			cout << "sample " << j << " belongs to cluster " << cluster_idx << endl;
			orb_features[i][cluster_idx]++;
		}
		// for(int j=0; j<orb_desc.rows; j++)
		// {
		//  // TODO find closest_center
		// 	orb_features[i][cluster_map[j]] /= (ipt_idx_end - ipt_idx_start);
		// }

	}

	ofstream feature_file;
	feature_file.open("orb_test_features_large.txt");
	for(int i=0; i<test_data_size; i++)
	{
		for(int j=0; j<k_clusters; j++)
		{
			feature_file << orb_features[i][j] << " ";
		}
		feature_file << endl;
	}
	feature_file.close();

}

void get_lbp_features(const Mat &lbp, float* features)
{
	for(int i=0; i<255; i++)
	{
		features[i] = 0;
	}
	for(int i=0; i<lbp.rows; i++)
	{
		const uchar* row_ptr = lbp.ptr<uchar>(i);
		for(int j=0; j<lbp.cols; j++)
		{
			features[row_ptr[j]]++;
		}
	}
	float total_pixels = lbp.rows*lbp.cols;
	for(int i=0; i<255; i++)
	{
		features[i] /= total_pixels;
	}
}

void _write_lbp_features(vector<string> data, string filename)
{
	// vector<string> test_data = get_test_data();
	int data_size = data.size();
	int pyramid[] = {1, 2, 4};

	int lbp_feature_count = 5355; // (1+4+16) * 255
	float** lbp_features = new float*[data_size];
	for(int x=0; x<data_size; x++)
	{
		lbp_features[x] = new float[lbp_feature_count];
		for(int y=0; y<lbp_feature_count; y++)
		{
			lbp_features[x][y] = 0;
		}
	}

	for(int i=0; i<data_size; i++)
	{
		Mat img = imread(data[i], CV_LOAD_IMAGE_GRAYSCALE);
		Mat lbp;
		findLBP(img, lbp);
		// get_lbp_features(lbp, lbp_features[i]);
		int offset_ptr = 0;
		for(int p=0; p<3; p++)
		{
			int dx = lbp.cols/pyramid[p];
			int dy = lbp.rows/pyramid[p];
			for(int x=0; x<pyramid[p]; x++)
			{
				for(int y=0; y<pyramid[p]; y++)
				{
					// cout << round(dx*x+dx/2.0) << ", " << round(dy*y+dy/2.0) << " -- size: " << dx << ", " << dy << endl;
					// int ix = round(dx*x+dx/2.0) + 1;
					// int iy = round(dy*y+dy/2.0) + 1;
					if(p==0)
					{
						get_lbp_features(lbp, lbp_features[i]+offset_ptr);
					}
					else
					{
						int ix = dx*x;
						int iy = dy*y;
						get_lbp_features(lbp(Rect(ix, iy, dx, dy)), lbp_features[i]+offset_ptr);
					}
					offset_ptr += 255;
					cout << "  " << offset_ptr << endl;
				}
			}
		}
		cout << i << endl;
	}

	ofstream feature_file;
	feature_file.open(filename);
	for(int i=0; i<data_size; i++)
	{
		for(int j=0; j<lbp_feature_count; j++)
		{
			feature_file << lbp_features[i][j] << " ";
		}
		feature_file << endl;
	}
	feature_file.close();
	//TODO: delete your memory you lazy bum
}

void write_lbp_training_features()
{
	vector<string> training_data = get_training_data();
	_write_lbp_features(training_data, "lbp_training_features_large.txt");
}

void write_lbp_test_features()
{
	vector<string> test_data = get_test_data();
	_write_lbp_features(test_data, "lbp_test_features_large.txt");
}

/*
g++ -O3 -std=c++0x extract_features.cpp -o extract_features `pkg-config --cflags --libs opencv`
*/
int main()
{
	write_orb_training_features();
	write_orb_test_features();
	write_lbp_training_features();
	write_lbp_test_features();
}