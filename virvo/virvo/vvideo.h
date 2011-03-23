//****************************************************************************
// Filename:          vvideo.h.cpp
// Author:            Michael Poehnl
// Institution:       University of Stuttgart, Supercomputing Center
// History:           08-01-2002  Creation date
//****************************************************************************

#ifndef _VVIDEO_H
#define _VVIDEO_H

/**This class is the interface to the XviD library (xvidcore-0.9.0)
   It is used by the vvImage class for the XviD encoding of RGB frames. <BR>

   @author Michael Poehnl
*/
#include "vvexport.h"

struct AVCodec;
struct AVCodecContext;
struct AVFrame;
struct AVPicture;

class VIRVOEXPORT vvideo
{
  public:
    vvideo( float fr=25.0f, int min_q=1, int max_q=31, int br=900000, int max_k=250);
    ~vvideo();
    int create_enc(int w, int h);
    int create_dec(int w, int h);
    int enc_frame(const unsigned char* src, unsigned char* dst, int* enc_size, int* key);
    int dec_frame(const unsigned char* src, unsigned char* dst, int src_size, int* dst_size);
    int del_enc();
    int del_dec();
    void set_framerate(float fr);
    void set_quantizer(int min_q, int max_q);
    void set_bitrate(int br);
    void set_max_key_interval(int max_k);

  private:
    float framerate;
    int min_quantizer;
    int max_quantizer;
    int bitrate;
    int max_key_interval;

    // ffmpeg
    AVCodec *encoder;
    AVCodec *decoder;
    AVCodecContext *enc_context;
    AVCodecContext *dec_context;
    AVFrame *enc_picture;
    AVFrame *dec_picture;
    AVPicture *yuv_picture;
    int codec_id;
    static bool global_avcodec_init_done;

    // XviD
    int stride;
    void *enc_handle;
    void *dec_handle;
};
#endif
