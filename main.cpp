#include <iostream>
 
using namespace std;
 
extern "C" {
#include <math.h>
#include <libavutil/opt.h>
#include <libavcodec/avcodec.h>
#include <libavutil/channel_layout.h>
#include <libavutil/common.h>
#include <libavutil/imgutils.h>
#include <libavutil/mathematics.h>
#include <libavutil/samplefmt.h>
#include <libavformat/avformat.h>
#include <libavdevice/avdevice.h>
#include <libavutil/dict.h>
#include <libswscale/swscale.h>
};

#include <boost/filesystem.hpp>

#include "yolo.hpp"
#include "utils.hpp"

namespace fs = boost::filesystem;
using namespace std;

static void detect(YOLO &net, vector<cv::Mat>& images,
                                      vector<string> names, TimeStamp *ts) {
  ts->save("detection overall");
  ts->save("stage 1: pre-process");
  net.preForward(images);
  ts->save("stage 1: pre-process");
  ts->save("stage 2: detection  ");
  net.forward();
  ts->save("stage 2: detection  ");
  ts->save("stage 3:post-process");
  vector<vector<yolov3_DetectRect>> dets = net.postForward();
  ts->save("stage 3:post-process");
  ts->save("detection overall");

  string save_folder = "result_imgs";
  if (!fs::exists(save_folder)) {
    fs::create_directory(save_folder);
  }

  for (size_t i = 0; i < images.size(); i++) {
    for (size_t j = 0; j < dets[i].size(); j++) {
      int x_min = dets[i][j].left;
      int x_max = dets[i][j].right;
      int y_min = dets[i][j].top;
      int y_max = dets[i][j].bot;

      std::cout << "Category: " << dets[i][j].category
        << " Score: " << dets[i][j].score << " : " << x_min <<
        "," << y_min << "," << x_max << "," << y_max << std::endl;

      cv::Rect rc;
      rc.x = x_min;
      rc.y = y_min;;
      rc.width = x_max - x_min;
      rc.height = y_max - y_min;
      cv::rectangle(images[i], rc, cv::Scalar(255, 0, 0), 2, 1, 0);
    }
    cv::imwrite(save_folder + "/" + names[i], images[i]);
  }
}

int main(int argc, char **argv) {
  int ret;
  const int nwidth = 1280; 
  const int nheight = 720;

  if (argc != 3 ) {
    cout << "USAGE:" << endl;
    cout << "  " << argv[0] << " <video name> <bmodel file> " << endl;
    exit(1);
  }

  char* video_name = argv[1];
  string bmodel_file = argv[2];

  if (!fs::exists(bmodel_file)) {
    cout << "Cannot find valid model file." << endl;
    exit(1);
  }

  AVFormatContext *fmtCtx = NULL;
  AVPacket pkt1, *packet = &pkt1;
  AVDictionary *options = NULL;

  avcodec_register_all();
  avdevice_register_all();

  AVInputFormat *inputFmt = av_find_input_format("video4linux2");
  if (NULL != inputFmt) {
    cout << "input format:" << inputFmt->name << endl;
  } else {
    cout << "error format!" << endl;
    exit(1);
  }

  av_dict_set(&options, "video_size", "1280x720", 0);
  ret = avformat_open_input(&fmtCtx, video_name, inputFmt, &options);
  if (0 == ret) {
    cout << "Open input device " << video_name << " success!" << endl;
  }

  if(avformat_find_stream_info(fmtCtx, NULL) < 0) {
    cout << "Couldn't find stream information" << endl;
    exit(1);
  }

  int videoindex = -1;
  for (size_t i = 0; i < fmtCtx->nb_streams; i++) {
    if(fmtCtx->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO) {
      videoindex = i;
    }
  }
  if (-1 == videoindex) {
    cout << "Couldn't find stream information" << endl;
    exit(1);
  }
  AVCodecContext* pCodecCtx = fmtCtx->streams[videoindex]->codec;

  AVFrame* pFrameRGB = av_frame_alloc();
  if (NULL == pFrameRGB) {
    exit(1);
  }
  AVFrame* pFrameYUYV = av_frame_alloc();
  if (NULL == pFrameYUYV) {
    exit(1);
  }
  struct SwsContext *img_convert_ctx;
  int numBytes = avpicture_get_size(AV_PIX_FMT_BGR24, nwidth, nheight);
  uint8_t* rgbBuffer = (uint8_t *)av_malloc(numBytes * sizeof(uint8_t));
  if (NULL == rgbBuffer) {
    exit(1);
  }
  img_convert_ctx = sws_getContext(nwidth, nheight, pCodecCtx->pix_fmt,
                   nwidth, nheight, AV_PIX_FMT_BGR24, SWS_BICUBIC, NULL, NULL, NULL);
  avpicture_fill((AVPicture*)pFrameRGB, rgbBuffer, AV_PIX_FMT_BGR24, nwidth, nheight);
  cv::Mat mat_img(cv::Size(nwidth, nheight), CV_8UC3, rgbBuffer);  

  YOLO net(bmodel_file);
  int id = 0;
  TimeStamp ts;
  net.enableProfile(&ts);

  while(1) {
    av_read_frame(fmtCtx, packet);
    avpicture_fill((AVPicture*)pFrameYUYV, packet->data, pCodecCtx->pix_fmt, nwidth, nheight);
    sws_scale(img_convert_ctx,
            (uint8_t const * const *) pFrameYUYV->data,
            pFrameYUYV->linesize, 0, nheight, pFrameRGB->data,
            pFrameRGB->linesize);

    vector<cv::Mat> imgs;
    vector<string> names;
    imgs.push_back(mat_img);
    names.push_back(to_string(id) + "_video.jpg");
    detect(net, imgs, names, &ts);
    id++;

    time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
    ts.calbr_basetime(base_time);
    ts.build_timeline("yolo detect");
    ts.show_summary("detect ");
    ts.clear();
  }

  av_frame_free(&pFrameYUYV);
  av_frame_free(&pFrameRGB);
  av_free(rgbBuffer);
  sws_freeContext(img_convert_ctx);

  av_free_packet(packet);
  avformat_close_input(&fmtCtx);
 
  return 0;
}
