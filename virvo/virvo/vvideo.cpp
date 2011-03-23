//****************************************************************************
// Filename:          vvideo.cpp
// Author:            Michael Poehnl
// Institution:       University of Stuttgart, Supercomputing Center
// History:           19-12-2002  Creation date
//****************************************************************************
#undef VV_FFMPEG

#include "vvideo.h"
#include "vvdebugmsg.h"
#ifndef NULL
#ifdef __GNUG__
#define NULL (__null)
#else
#define NULL (0)
#endif
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifdef VV_FFMPEG
#include <ffmpeg/avcodec.h>

bool vvideo::global_avcodec_init_done = false;
#elif defined(VV_XVID)
#include <xvid.h>
#endif

#if !defined(VV_FFMPEG) && defined(VV_XVID)
//----------------------------------------------------------------------------
/** Destructor
 */
static int const motion_presets[7] =
{
  0,                                              // Q 0
  PMV_EARLYSTOP16,                                // Q 1
  PMV_EARLYSTOP16,                                // Q 2
  PMV_EARLYSTOP16 | PMV_HALFPELREFINE16,          // Q 3
  PMV_EARLYSTOP16 | PMV_HALFPELREFINE16,          // Q 4
                                                  // Q 5
  PMV_EARLYSTOP16 | PMV_HALFPELREFINE16 | PMV_EARLYSTOP8 |
  PMV_HALFPELREFINE8,
                                                  // Q 6
  PMV_EARLYSTOP16 | PMV_HALFPELREFINE16 | PMV_EXTSEARCH16 |
  PMV_USESQUARES16 | PMV_EARLYSTOP8 | PMV_HALFPELREFINE8
};

//----------------------------------------------------------------------------
/** Destructor
 */
static int const general_presets[7] =
{
  XVID_H263QUANT,                                 // Q 0
  XVID_MPEGQUANT,                                 // Q 1
  XVID_H263QUANT,                                 // Q 2
  XVID_H263QUANT | XVID_HALFPEL,                  // Q 3
  XVID_H263QUANT | XVID_HALFPEL | XVID_INTER4V,   // Q 4
  XVID_H263QUANT | XVID_HALFPEL | XVID_INTER4V,   // Q 5
  XVID_H263QUANT | XVID_HALFPEL | XVID_INTER4V    // Q 6
};
#endif

//----------------------------------------------------------------------------
/** Constructor
@param fr  framerate
@param min_q  lower bound for quantizer
@param max_q  upper bound for quantizer
@param br  target bitrate
@param max_k  maximum key interval
*/
vvideo::vvideo(float fr, int min_q, int max_q, int br, int max_k)
:framerate(fr), min_quantizer(min_q), max_quantizer(max_q), bitrate(br), max_key_interval(max_k)
#if defined(VV_FFMPEG)
, encoder(NULL), decoder(NULL),
enc_context(NULL), dec_context(NULL),
enc_picture(NULL), dec_picture(NULL),
yuv_picture(NULL)
#elif defined(VV_XVID)
, stride(0),
enc_handle(NULL), dec_handle(NULL)
#endif
{
#if defined(VV_FFMPEG)
  if(!global_avcodec_init_done)
  {
    avcodec_init();
    avcodec_register_all();
    global_avcodec_init_done = true;
  }
  codec_id = (int)CODEC_ID_MPEG1VIDEO;
  codec_id = (int)CODEC_ID_RAWVIDEO;
  codec_id = (int)CODEC_ID_MJPEG;
#endif
}

//----------------------------------------------------------------------------
/** Destructor
 */
vvideo::~vvideo()
{
  del_enc();
  del_dec();
}

//----------------------------------------------------------------------------
/** Creates an XviD encoder
@param w  width of frames
@param h  height of frames
@return   0 for success, != 0 for error
*/
int vvideo::create_enc(int w, int h)
{
  del_enc();

#if defined(VV_FFMPEG)
  encoder = avcodec_find_encoder((enum CodecID)codec_id);
  if(!encoder)
  {
    vvDebugMsg::msg(1, "error: failed to find encoder");
    return -1;
  }

  enc_context = avcodec_alloc_context();
  if(!enc_context)
  {
    vvDebugMsg::msg(1, "error: failed to allocate encoding context");
    return -1;
  }
  enc_context->bit_rate = bitrate;
  enc_context->width = w;
  enc_context->height = h;
  if(fabs(framerate - (int)framerate) < 0.001)
  {
    enc_context->frame_rate = (int)framerate;
    enc_context->frame_rate_base = 1;
  }
  else
  {
    enc_context->frame_rate = (int)(framerate*1000.0);
    enc_context->frame_rate_base = 1000;
  }
  enc_context->gop_size = max_key_interval;
  enc_context->max_b_frames = 0;                  // XXX: wie viele? style-abhaengig?
  enc_context->pix_fmt = PIX_FMT_RGB24;
  //enc_context->pix_fmt = PIX_FMT_YUV420P;
  enc_context->qmin = min_quantizer;
  enc_context->qmax = max_quantizer;
  enc_context->flags |= CODEC_FLAG_LOW_DELAY;

  if(avcodec_open(enc_context, encoder) < 0)
  {
    vvDebugMsg::msg(0, "error: failed to open encoder");
    return -1;
  }

  enc_picture = avcodec_alloc_frame();
  if(!enc_picture)
  {
    vvDebugMsg::msg(1, "error: failed to allocate encoding picture");
    return -1;
  }
  int size = avpicture_get_size(enc_context->pix_fmt, w, h);
  fprintf(stderr, "enc: size=%d, fmt=%d\n", size, enc_context->pix_fmt);
  uint8_t *picture_buf = (uint8_t *)malloc(size);
  avpicture_fill((AVPicture *)enc_picture, picture_buf, enc_context->pix_fmt, w, h);

  yuv_picture = new struct AVPicture;
  yuv_picture->data[0] = new uint8_t[w*h];
  yuv_picture->data[1] = new uint8_t[w*h/4];
  yuv_picture->data[2] = new uint8_t[w*h/4];
  yuv_picture->data[3] = NULL;
  yuv_picture->linesize[0] = w;
  yuv_picture->linesize[1] = w/2;
  yuv_picture->linesize[2] = w/2;
  yuv_picture->linesize[3] = 0;

  return 0;
#elif defined(VV_XVID)
  int xerr;
  XVID_INIT_PARAM xinit;
  XVID_ENC_PARAM xparam;

  xinit.cpu_flags = XVID_CPU_FORCE;
  xvid_init(0, 0, &xinit, 0);
  if (xinit.api_version != API_VERSION)
  {
    vvDebugMsg::msg(1,"Wrong Xvid library version");
    return -1;
  }
  xparam.width = w;
  xparam.height = h;
  if ((framerate - (int)framerate) < 0.001)
  {
    xparam.fincr = 1;
    xparam.fbase = (int)framerate;
  }
  else
  {
    xparam.fincr = 1000;
    xparam.fbase = (int)(1000 * framerate);
  }
  xparam.rc_reaction_delay_factor = -1;           //default values
  xparam.rc_averaging_period = -1;                //default values
  xparam.rc_buffer = -1;                          //default values
  xparam.rc_bitrate = bitrate;
  xparam.min_quantizer = min_quantizer;
  xparam.max_quantizer = max_quantizer;
  xparam.max_key_interval = max_key_interval;
  xerr = xvid_encore(0, XVID_ENC_CREATE, &xparam, 0);
  enc_handle=xparam.handle;
  if (xerr == 0)
    vvDebugMsg::msg(3, "XviD Encoder created");
  return xerr;
#else
  (void)w;
  (void)h;
  return -1;
#endif
}

//----------------------------------------------------------------------------
/** Creates an XviD decoder
@param w  width of frames
@param h  height of frames
@return   0 for success, != 0 for error
*/
int vvideo::create_dec(int w, int h)
{
  del_dec();

#if defined(VV_FFMPEG)
  decoder = avcodec_find_decoder((enum CodecID)codec_id);
  if(!decoder)
  {
    vvDebugMsg::msg(1, "error: failed to find decoder");
    return -1;
  }

  dec_context = avcodec_alloc_context();
  if(!dec_context)
  {
    vvDebugMsg::msg(1, "error: failed to allocate decoding context");
    return -1;
  }
  dec_context->width = w;
  dec_context->height = h;
  dec_context->pix_fmt = PIX_FMT_RGB24;
  //dec_context->pix_fmt = PIX_FMT_YUV420P;
  dec_context->flags |= CODEC_FLAG_LOW_DELAY;

  if(avcodec_open(dec_context, decoder) < 0)
  {
    vvDebugMsg::msg(0, "error: failed to open decoder");
    return -1;
  }
  int size=0;
  fprintf(stderr, "deca: size=%d, fmt=%d\n", size, dec_context->pix_fmt);

  dec_picture = avcodec_alloc_frame();
  if(!dec_picture)
  {
    vvDebugMsg::msg(1, "error: failed to allocate decoding picture");
    return -1;
  }
  fprintf(stderr, "decb: size=%d, fmt=%d\n", size, dec_context->pix_fmt);
  size = avpicture_get_size(dec_context->pix_fmt, w, h);
  fprintf(stderr, "decc: size=%d, fmt=%d\n", size, dec_context->pix_fmt);
  uint8_t *picture_buf = (uint8_t *)malloc(size);
  avpicture_fill((AVPicture *)dec_picture, picture_buf, dec_context->pix_fmt, w, h);
  fprintf(stderr, "decd: size=%d, fmt=%d\n", size, dec_context->pix_fmt);

  return 0;
#elif defined(VV_XVID)
  int xerr;
  XVID_INIT_PARAM xinit;
  XVID_DEC_PARAM xparam;

  xinit.cpu_flags = XVID_CPU_FORCE;
  xinit.cpu_flags = 0;
  xvid_init(NULL, 0, &xinit, NULL);
  if (xinit.api_version != API_VERSION)
  {
    vvDebugMsg::msg(1,"Wrong Xvid library version");
    return -1;
  }
  xparam.width = w;
  stride = w;
  xparam.height = h;
  xerr = xvid_decore(NULL, XVID_DEC_CREATE, &xparam, NULL);
  dec_handle = xparam.handle;
  if (xerr == 0)
    vvDebugMsg::msg(3, "XviD Decoder created");
  return xerr;
#else
  (void)w;
  (void)h;
  return -1;
#endif
}

//----------------------------------------------------------------------------
/** Encodes a frame
@param src  pointer to frame to encode
@param dst  pointer to destination
@param enc_size  IN, available space for encoded frame
@param enc_size  OUT, size of encoded frame
@param key  OUT, encoded frame is a keyframe if != 0
@param style  style of encoding [0 .. 6]
@param quant  quantizer value to use for that frame [1 .. 31]
@return   0 for success, != 0 for error
*/
int vvideo::enc_frame(const unsigned char* src, unsigned char* dst, int* enc_size, int* key)
{
#if defined(VV_FFMPEG)
#if 0
  static int fc = 0;
  fc = (fc+1)%25;
  AVPicture src_picture;
  src_picture.data[0] = (unsigned char *)src;
  src_picture.data[1] = src_picture.data[2] = src_picture.data[3] = NULL;
  src_picture.linesize[0] = enc_context->width*3;

  enc_picture->linesize[0] = enc_context->width;
  enc_picture->linesize[1] = enc_picture->linesize[2] = enc_context->width/2;
  img_convert(yuv_picture, PIX_FMT_YUV420P,
    &src_picture, PIX_FMT_RGB24, enc_context->width, enc_context->height);
  for(int i=0; i<4; i++)
  {
    enc_picture->linesize[i] = yuv_picture->linesize[i];
    enc_picture->data[i] = yuv_picture->data[i];
    //fprintf(stderr, "linesize[%d]=%d\n", i, yuv_picture->linesize[i]);
  }
#else
  for(int i=0; i<4; i++)
  {
    enc_picture->linesize[i] = 0;
    enc_picture->data[i] = (uint8_t *)src;
    //fprintf(stderr, "linesize[%d]=%d\n", i, yuv_picture->linesize[i]);
  }
  enc_picture->linesize[0] = enc_context->width*3;
  enc_picture->data[0] = (uint8_t *)src;
#endif
  *enc_size = avcodec_encode_video(enc_context, dst, *enc_size, enc_picture);

#if 0
  for(int y=0;y<enc_context->height;y++)
    for(int x=0;x<enc_context->width;x++)
      yuv_picture->data[0][y * yuv_picture->linesize[0] + x] = x + y + fc * 3;

  /* Cb and Cr */
  for(int y=0;y<enc_context->height/2;y++)
    for(int x=0;x<enc_context->width/2;x++)
  {
    yuv_picture->data[1][y * yuv_picture->linesize[1] + x] = 128 + y + fc * 2;
    yuv_picture->data[2][y * yuv_picture->linesize[2] + x] = 64 + x + fc * 5;
  }
#endif
#if 0
  for(int y=0;y<enc_context->height;y++)
    for(int x=0;x<enc_context->width;x++)
      yuv_picture->data[0][y * yuv_picture->linesize[0] + x] = x<enc_context->height/2?255:0;

  /* Cb and Cr */
  for(int y=0;y<enc_context->height/2;y++)
    for(int x=0;x<enc_context->width/2;x++)
  {
    yuv_picture->data[1][y * yuv_picture->linesize[1] + x] = y<enc_context->width/4?0:127;
    yuv_picture->data[2][y * yuv_picture->linesize[2] + x] = 0;
  }
#endif

  if(enc_picture->key_frame)
    *key = 1;
  else
    *key = 0;

  fprintf(stderr, "enc_size=%d, keyframe=%d\n", *enc_size, *key);

  return 0;
#elif defined(VV_XVID)
  int xerr;
  XVID_ENC_FRAME xframe;
  int style = 0;
  int quant = 1;

  if (style < 0 || style > 6)
    style = 0;
  if (quant < 1 || quant > 31)
    quant = 1;
  xframe.bitstream = dst;
  xframe.length = -1;                             // this is written by the routine
  xframe.image = (void *)src;
  xframe.colorspace = XVID_CSP_RGB24;
  xframe.intra = -1;
  xframe.quant = quant;
  xframe.motion = motion_presets[style];
  xframe.general = general_presets[style];
  xframe.quant_intra_matrix = xframe.quant_inter_matrix = 0;
  xerr = xvid_encore(enc_handle, XVID_ENC_ENCODE, &xframe, 0);
  *key = xframe.intra;
  *enc_size = xframe.length;
  if (xerr == 0)
    vvDebugMsg::msg(3, "frame encoded in XviD style");
  return xerr;
#else
  (void)src;
  (void)dst;
  (void)enc_size;
  (void)key;
  return -1;
#endif
}

//----------------------------------------------------------------------------
/** Decodes a frame
@param src  pointer to encoded frame
@param dst  pointer to destination
@param src_size  size of encoded frame
@param dst_size  IN, available space for decoded frame
@param dst_size  OUT, size of decoded frame
@return   0 for success, != 0 for error
*/
int vvideo::dec_frame(const unsigned char* src, unsigned char* dst, int src_size, int* dst_size)
{
#if defined(VV_FFMPEG)
  int got_picture;
#if 0
  *dst_size = avcodec_decode_video(dec_context, dec_picture, &got_picture, (unsigned char *)src, src_size);
#else
  *dst_size = avcodec_decode_video(dec_context, dec_picture, &got_picture, (unsigned char *)src, src_size);
  memcpy(dst, dec_picture->data[0], dec_context->width*dec_context->height*3);
#endif
  if(*dst_size != src_size)
    fprintf(stderr, "src_size=%d, dst_size=%d\n", src_size, *dst_size);
  *dst_size = dec_context->width * dec_context->height * 3;

  if(!got_picture)
  {
    fprintf(stderr, "no picture: src_size=%d\n");
    return -1;
  }
#if 0

  AVPicture dst_picture;
  dst_picture.data[0] = dst;
  dst_picture.data[1] = dst_picture.data[2] = dst_picture.data[3] = NULL;
  AVPicture src_picture;
  for(int i=0; i<4; i++)
  {
    src_picture.data[i] = dec_picture->data[i];
    src_picture.linesize[i] = dec_picture->linesize[i];
    fprintf(stderr, "linesize[%d]=%d\n", i, dec_picture->linesize[i]);
  }
  src_picture.linesize[0] = dec_context->width;
  src_picture.linesize[1] = dec_context->width/2;
  src_picture.linesize[2] = dec_context->width/2;

  dst_picture.linesize[0] = dst_picture.linesize[1] = dst_picture.linesize[3] = dec_context->width;
  fprintf(stderr, "w=%d, h=%d\n", dec_context->width, dec_context->height);
#if 0
  img_convert(&dst_picture, PIX_FMT_RGB24,
    &src_picture, PIX_FMT_YUV420P,
    dec_context->width, dec_context->height);
#else
  memset(dst, '\0', 3*dec_context->width*dec_context->height);
  memcpy(dst, dec_picture->data[0], 3*dec_context->width*dec_context->height/2);
  memcpy(dst, dec_picture->data[0], 3*dec_context->width*dec_context->height);
#endif
#endif

  return 0;
#elif defined(VV_XVID)
  int xerr;
  XVID_DEC_FRAME xframe;

  xframe.bitstream = (void *)src;
  xframe.length = src_size;
  xframe.stride = stride;
  xframe.image = dst;
  xframe.colorspace = XVID_CSP_RGB24 | XVID_CSP_VFLIP;
  xerr = xvid_decore(dec_handle, XVID_DEC_DECODE, &xframe, 0);
  *dst_size =  xframe.length;
  if (xerr == 0)
    vvDebugMsg::msg(3, "XviD frame decoded");
  return xerr;
#else
  (void)src;
  (void)dst;
  (void)src_size;
  (void)dst_size;
  return -1;
#endif
}

//----------------------------------------------------------------------------
/** Sets the target framerate
@param fr  framerate
*/
void vvideo::set_framerate(float fr)
{
  framerate = fr;
}

//----------------------------------------------------------------------------
/** Sets the quantizer bounds
@param min_q  lower quantizer bound
@param max_q  upper quantizer bound
*/
void vvideo::set_quantizer(int min_q, int max_q)
{
  min_quantizer = min_q;
  max_quantizer = max_q;
}

//----------------------------------------------------------------------------
/** Sets the target bitrate
@param br  bitrate
*/
void vvideo::set_bitrate(int br)
{
  bitrate = br;
}

//----------------------------------------------------------------------------
/** Sets the maximum interval for key frames
@param max_k maximum interval in frames
*/
void vvideo::set_max_key_interval(int max_k)
{
  max_key_interval = max_k;
}

//----------------------------------------------------------------------------
/** Deletes the encoder
@return   0 for success, != 0 for error
*/
int vvideo::del_enc()
{
#if defined(VV_FFMPEG)
  if(yuv_picture)
  {
    for(int i=0; i<4; i++)
      delete[] yuv_picture->data[i];
    delete yuv_picture;
    yuv_picture = NULL;
  }

  if(enc_context)
  {
    avcodec_close(enc_context);
    free(enc_context);
    enc_context = NULL;
  }
  encoder = NULL;

  if(enc_picture)
  {
    free(enc_picture);
    enc_picture = NULL;
  }

  return 0;
#elif defined(VV_XVID)
  int xerr = 0;

  if(enc_handle)
    xerr = xvid_encore(enc_handle, XVID_ENC_DESTROY, 0, 0);
  enc_handle = 0;
  return xerr;
#else
  return -1;
#endif
}

//----------------------------------------------------------------------------
/** Deletes the decoder
@return   0 for success, != 0 for error
*/
int vvideo::del_dec()
{
#if defined(VV_FFMPEG)
  if(dec_context)
  {
    avcodec_close(dec_context);
    free(dec_context);
    dec_context = NULL;
  }
  decoder = NULL;

  if(dec_picture)
  {
    free(dec_picture);
    dec_picture = NULL;
  }

  return 0;
#elif defined(VV_XVID)
  int xerr = 0;

  if(dec_handle)
    xerr = xvid_decore(dec_handle, XVID_DEC_DESTROY, 0, 0);
  dec_handle = 0;
  return xerr;
#else
  return -1;
#endif
}
