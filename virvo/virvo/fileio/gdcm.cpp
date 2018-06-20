#include "gdcm.h"

#if VV_HAVE_GDCM
#include "gdcmAttribute.h"
#include "gdcmReader.h"
#include "gdcmImageReader.h"
#include "gdcmMediaStorage.h"
#include "gdcmFile.h"
#include "gdcmDataSet.h"
#include "gdcmUIDs.h"
#include "gdcmGlobal.h"
#include "gdcmModules.h"
#include "gdcmDefs.h"
#include "gdcmOrientation.h"
#include "gdcmVersion.h"
#include "gdcmMD5.h"
#include "gdcmSystem.h"
#include "gdcmDirectory.h"

#include <virvo/vvvoldesc.h>
#include <virvo/vvpixelformat.h>
#include <virvo/vvfileio.h>
#include "exceptions.h"

namespace {

void load_dicom_image(vvVolDesc *vd, virvo::gdcm::dicom_meta &meta)
{
  bool subfile = false;

  gdcm::ImageReader reader;
  reader.SetFileName( vd->getFilename() );
  if( !reader.Read() )
  {
    std::cerr << "Could not read image from: " << vd->getFilename() << std::endl;
    throw virvo::fileio::exception("read error");
  }
  const gdcm::File &file = reader.GetFile();
  const gdcm::DataSet &ds = file.GetDataSet();
  const gdcm::Image &image = reader.GetImage();
  const double *dircos = image.GetDirectionCosines();
  gdcm::Orientation::OrientationType type = gdcm::Orientation::GetType(dircos);
  const char *label = gdcm::Orientation::GetLabel( type );
  //image.Print( std::cerr );
  if (!subfile)
    std::cerr << "Orientation Label: " << label << std::endl;
  bool lossy = image.IsLossy();
  if (!subfile)
    std::cerr << "Encapsulated Stream was found to be: " << (lossy ? "lossy" : "lossless") << std::endl;

  const unsigned int *dim = image.GetDimensions();
  vd->vox[0] = dim[0];
  vd->vox[1] = dim[1];
  vd->vox[2] = 1;
  const double *spacing = image.GetSpacing();
  vd->setDist(static_cast<float>(spacing[0]),
      static_cast<float>(spacing[1]),
      static_cast<float>(spacing[2]));
  gdcm::PixelFormat pf = image.GetPixelFormat();
  switch(pf.GetBitsAllocated()/8)
  {
  case 1:
  case 2:
    vd->bpc = pf.GetBitsAllocated()/8;
    vd->setChan(1);
    break;
  case 3:
  case 4:
    vd->bpc = 1;
    vd->setChan(pf.GetBitsAllocated()/8);
    break;
  default: assert(0); break;
  }

  gdcm::Attribute<0x0020,0x0011> attrSequenceNumber;
  attrSequenceNumber.Set(ds);
  meta.sequence = attrSequenceNumber.GetValue();

  gdcm::Attribute<0x0020,0x0013> attrImageNumber;
  attrImageNumber.Set(ds);
  meta.slice = attrImageNumber.GetValue();

  gdcm::Attribute<0x0020,0x1041> attrSliceLocation;
  attrSliceLocation.Set(ds);
  meta.spos = attrSliceLocation.GetValue();

  char *rawData = new char[image.GetBufferLength()];
  image.GetBuffer(rawData);
  vd->addFrame((uint8_t *)rawData, vvVolDesc::ARRAY_DELETE, meta.slice);
  ++vd->frames;

  meta.slope = static_cast<float>(image.GetSlope());
  meta.intercept = static_cast<float>(image.GetIntercept());

  meta.format = virvo::PF_R8;
  if (pf == gdcm::PixelFormat::INT16)
  {
    meta.format = virvo::PF_R16I;
  }
  else if (pf == gdcm::PixelFormat::INT32)
  {
    meta.format = virvo::PF_R32I;
  }
  else if (pf == gdcm::PixelFormat::UINT32)
  {
    meta.format = virvo::PF_R32UI;
  }
}


void load_dicom_dir(vvVolDesc *vd, gdcm::Reader &reader, virvo::gdcm::dicom_meta &meta)
{
  typedef std::set<gdcm::DataElement> DataElementSet;
  typedef DataElementSet::const_iterator ConstIterator;

  gdcm::File &file = reader.GetFile();
  gdcm::DataSet &ds = file.GetDataSet();
  gdcm::FileMetaInformation &fmi = file.GetHeader();
  // copied straight from GDCM: Examples/Cxx/ReadAndDumpDICOMDIR.cxx
  std::stringstream strm;

  if (fmi.FindDataElement( gdcm::Tag (0x0002, 0x0002)))
  {   strm.str("");
    fmi.GetDataElement( gdcm::Tag (0x0002, 0x0002) ).GetValue().Print(strm);
  }
  else
  {
    std::cerr << " Media Storage Sop Class UID not present" << std::endl;
  }

  //TODO il faut trimer strm.str() avant la comparaison au cas ou...
  if ("1.2.840.10008.1.3.10"!=strm.str())
  {
    throw virvo::fileio::exception("format error: not a DICOMDIR");
  }

  int maxNumImg = 0;
  int maxItem = -1;
  int numSeq = 0;
  int readSeq = -1;
  int wantSeq = vd->getEntry();

  ConstIterator it = ds.GetDES().begin();
  ConstIterator maxDataSet = ds.GetDES().end();

  for( ; it != ds.GetDES().end(); ++it)
  {

    if (it->GetTag()==gdcm::Tag (0x0004, 0x1220))
    {

      const gdcm::DataElement &de = (*it);
      // ne pas utiliser GetSequenceOfItems pour extraire les items
      gdcm::SmartPointer<gdcm::SequenceOfItems> sqi =de.GetValueAsSQ();
      unsigned int itemused = 1;
      std::string patient;
      while (itemused<=sqi->GetNumberOfItems())

      {
        strm.str("");

        if (sqi->GetItem(itemused).FindDataElement(gdcm::Tag (0x0004, 0x1430)))
          sqi->GetItem(itemused).GetDataElement(gdcm::Tag (0x0004, 0x1430)).GetValue().Print(strm);

        //TODO il faut trimer strm.str() avant la comparaison
        while((strm.str()=="PATIENT")||((strm.str()=="PATIENT ")))
        {
          //std::cerr << strm.str() << std::endl;
          strm.str("");
          if (sqi->GetItem(itemused).FindDataElement(gdcm::Tag (0x0010, 0x0010)))
            sqi->GetItem(itemused).GetDataElement(gdcm::Tag (0x0010, 0x0010)).GetValue().Print(strm);
          //std::cerr << "PATIENT NAME : " << strm.str() << std::endl;
          patient = strm.str();


          //PATIENT ID
          strm.str("");
          if (sqi->GetItem(itemused).FindDataElement(gdcm::Tag (0x0010, 0x0020)))
            sqi->GetItem(itemused).GetDataElement(gdcm::Tag (0x0010, 0x0020)).GetValue().Print(strm);
          //std::cerr << "PATIENT ID : " << strm.str() << std::endl;

          /*ADD TAG TO READ HERE*/
          //std::cerr << "=========================== "  << std::endl;
          itemused++;
          strm.str("");
          if (sqi->GetItem(itemused).FindDataElement(gdcm::Tag (0x0004, 0x1430)))
            sqi->GetItem(itemused).GetDataElement(gdcm::Tag (0x0004, 0x1430)).GetValue().Print(strm);

          std::string study;
          //TODO il faut trimer strm.str() avant la comparaison
          while((strm.str()=="STUDY")||((strm.str()=="STUDY ")))
          {
            //std::cerr << "  " << strm.str() << std::endl;
            //UID
            strm.str("");
            if (sqi->GetItem(itemused).FindDataElement(gdcm::Tag (0x0020, 0x000d)))
              sqi->GetItem(itemused).GetDataElement(gdcm::Tag (0x0020, 0x000d)).GetValue().Print(strm);
            //std::cerr << "      STUDY UID : " << strm.str() << std::endl;

            //STUDY DATE
            strm.str("");
            if (sqi->GetItem(itemused).FindDataElement(gdcm::Tag (0x0008, 0x0020)))
              sqi->GetItem(itemused).GetDataElement(gdcm::Tag (0x0008, 0x0020)).GetValue().Print(strm);
            //std::cerr << "      STUDY DATE : " << strm.str() << std::endl;

            //STUDY DESCRIPTION
            strm.str("");
            if (sqi->GetItem(itemused).FindDataElement(gdcm::Tag (0x0008, 0x1030)))
              sqi->GetItem(itemused).GetDataElement(gdcm::Tag (0x0008, 0x1030)).GetValue().Print(strm);
            //std::cerr << "      STUDY DESCRIPTION : " << strm.str() << std::endl;
            study = strm.str();

            /*ADD TAG TO READ HERE*/
            //std::cerr << "      " << "=========================== "  << std::endl;

            itemused++;
            strm.str("");
            if (sqi->GetItem(itemused).FindDataElement(gdcm::Tag (0x0004, 0x1430)))
              sqi->GetItem(itemused).GetDataElement(gdcm::Tag (0x0004, 0x1430)).GetValue().Print(strm);

            std::string series;
            //TODO il faut trimer strm.str() avant la comparaison
            while((strm.str()=="SERIES")||((strm.str()=="SERIES ")))
            {
              //std::cerr << "      " << strm.str() << std::endl;
              strm.str("");
              if (sqi->GetItem(itemused).FindDataElement(gdcm::Tag (0x0020, 0x000e)))
                sqi->GetItem(itemused).GetDataElement(gdcm::Tag (0x0020, 0x000e)).GetValue().Print(strm);
              //std::cerr << "          SERIE UID" << strm.str() << std::endl;

              //SERIE MODALITY
              strm.str("");
              if (sqi->GetItem(itemused).FindDataElement(gdcm::Tag (0x0008, 0x0060)))
                sqi->GetItem(itemused).GetDataElement(gdcm::Tag (0x0008, 0x0060)).GetValue().Print(strm);
              //std::cerr << "          SERIE MODALITY" << strm.str() << std::endl;

              //SERIE DESCRIPTION
              strm.str("");
              if (sqi->GetItem(itemused).FindDataElement(gdcm::Tag (0x0008, 0x103e)))
                sqi->GetItem(itemused).GetDataElement(gdcm::Tag (0x0008, 0x103e)).GetValue().Print(strm);
              //std::cerr << "          SERIE DESCRIPTION" << strm.str() << std::endl;
              series = strm.str();


              /*ADD TAG TO READ HERE*/

              //std::cerr << "          " << "=========================== "  << std::endl;
              itemused++;
              strm.str("");
              if (sqi->GetItem(itemused).FindDataElement(gdcm::Tag (0x0004, 0x1430)))
                sqi->GetItem(itemused).GetDataElement(gdcm::Tag (0x0004, 0x1430)).GetValue().Print(strm);


              int curitem = itemused;
              int numimg = 0;
              //TODO il faut trimer strm.str() avant la comparaison
              while ((strm.str()=="IMAGE")||((strm.str()=="IMAGE ")))
                // if(tmp=="IMAGE")
              {
                ++numimg;
                //std::cerr << "          " << strm.str() << std::endl;

                sqi->GetItem(itemused);


                //UID
                strm.str("");
                if (sqi->GetItem(itemused).FindDataElement(gdcm::Tag (0x0004, 0x1511)))
                  sqi->GetItem(itemused).GetDataElement(gdcm::Tag (0x0004, 0x1511)).GetValue().Print(strm);
                //std::cerr << "              IMAGE UID : " << strm.str() << std::endl;

                //PATH de l'image
                strm.str("");
                if (sqi->GetItem(itemused).FindDataElement(gdcm::Tag (0x0004, 0x1500)))
                  sqi->GetItem(itemused).GetDataElement(gdcm::Tag (0x0004, 0x1500)).GetValue().Print(strm);
                //std::cerr << "              IMAGE PATH : " << strm.str() << std::endl;
                /*ADD TAG TO READ HERE*/

                // sequence number of image
                int seq = -1;
                strm.str("");
                if (sqi->GetItem(itemused).FindDataElement(gdcm::Tag (0x0020, 0x0011)))
                {
                  sqi->GetItem(itemused).GetDataElement(gdcm::Tag (0x0020, 0x0011)).GetValue().Print(strm);
                  std::stringstream s(strm.str());
                  s >> seq;
                }
                //std::cerr << "              SEQUENCE NUMBER : " << strm.str() << std::endl;

                // number of image in sequence
                int img = -1;
                strm.str("");
                if (sqi->GetItem(itemused).FindDataElement(gdcm::Tag (0x0020, 0x0013)))
                {
                  sqi->GetItem(itemused).GetDataElement(gdcm::Tag (0x0020, 0x0013)).GetValue().Print(strm);
                  std::stringstream s(strm.str());
                  s >> img;
                }
                //std::cerr << "              IMAGE NUMBER : " << strm.str() << std::endl;

                if(itemused < sqi->GetNumberOfItems())
                {
                  itemused++;
                }
                else
                {
                  break;
                }
                std::cerr << "img=" << img << ", seq=" << seq << std::endl;

                strm.str("");

                if (sqi->GetItem(itemused).FindDataElement(gdcm::Tag (0x0004, 0x1430)))
                  sqi->GetItem(itemused).GetDataElement(gdcm::Tag (0x0004, 0x1430)).GetValue().Print(strm);

              }

              std::cerr << "Sequence " << numSeq << " with " << numimg << " images (" << patient << "/" << study << "/" << series << ")" << std::endl;

              if ((wantSeq<0 && numimg>maxNumImg) || (wantSeq>=0 && numSeq==wantSeq))
              {
                readSeq = numSeq;
                maxNumImg = numimg;
                maxItem = curitem;
                maxDataSet = it;
              }

              ++numSeq;
            }
          }
        }
        itemused++;
      }
    }
  }
  std::cerr << "Found " << numSeq << " image sequences, reading " << readSeq << std::endl;

  std::vector<std::string> filelist;

  const gdcm::DataElement &de = *maxDataSet;
  // ne pas utiliser GetSequenceOfItems pour extraire les items
  gdcm::SmartPointer<gdcm::SequenceOfItems> sqi =de.GetValueAsSQ();
  unsigned int itemused = maxItem;

  strm.str("");
  if (sqi->GetItem(itemused).FindDataElement(gdcm::Tag (0x0004, 0x1430)))
    sqi->GetItem(itemused).GetDataElement(gdcm::Tag (0x0004, 0x1430)).GetValue().Print(strm);

  //TODO il faut trimer strm.str() avant la comparaison
  while ((strm.str()=="IMAGE")||((strm.str()=="IMAGE ")))
    // if(tmp=="IMAGE")
  {
    //std::cerr << "          " << strm.str() << std::endl;


    //UID
    strm.str("");
    if (sqi->GetItem(itemused).FindDataElement(gdcm::Tag (0x0004, 0x1511)))
      sqi->GetItem(itemused).GetDataElement(gdcm::Tag (0x0004, 0x1511)).GetValue().Print(strm);
    //std::cerr << "              IMAGE UID : " << strm.str() << std::endl;

    //PATH de l'image
    strm.str("");
    if (sqi->GetItem(itemused).FindDataElement(gdcm::Tag (0x0004, 0x1500)))
      sqi->GetItem(itemused).GetDataElement(gdcm::Tag (0x0004, 0x1500)).GetValue().Print(strm);
    //std::cerr << "              IMAGE PATH : " << strm.str() << std::endl;

    std::string filename = vd->getFilename();
    std::replace(filename.begin(), filename.end(), '\\', '/');
    if (filename.find('/') == std::string::npos)
      filename.clear();
    else
      filename = vvToolshed::extractDirname(filename);
    if (filename.length()>0)
    {
      if (filename[filename.length()-1] != '/')
        filename += "/";
    }
    std::string s = strm.str();
    s = vvToolshed::strTrim(s);
    std::replace(s.begin(), s.end(), '\\', '/');
    filename += s;

    filelist.push_back(filename);

    /*ADD TAG TO READ HERE*/


    if(itemused < sqi->GetNumberOfItems())
    {itemused++;
    }else{break;}

    strm.str("");

    if (sqi->GetItem(itemused).FindDataElement(gdcm::Tag (0x0004, 0x1430)))
      sqi->GetItem(itemused).GetDataElement(gdcm::Tag (0x0004, 0x1430)).GetValue().Print(strm);
  }

  bool first = true;
  for (std::vector<std::string>::iterator it=filelist.begin();
       it != filelist.end();
       ++it)
  {
    const std::string &filename = *it;
    vvVolDesc *newVD = new vvVolDesc(filename.c_str());
    try
    {
      virvo::gdcm::dicom_meta nmeta;
      load_dicom_image(newVD, nmeta);
      if (first)
      {
          meta = nmeta;
      }
      else
      {
        if (meta.format != nmeta.format)
          throw virvo::fileio::exception("format error: slice formats do not match");
        if (meta.slope != nmeta.slope)
          throw virvo::fileio::exception("format error: slice slopes do not match");
        if (meta.intercept != nmeta.intercept)
          throw virvo::fileio::exception("format error: slice intercepts do not match");
      }
    }
    catch (...)
    {
      delete newVD;
      std::cerr << "vvFileIO::loadDicomFile: failed to load " << filename << std::endl;
      throw;
    }

    vvVolDesc::ErrorType mergeErr = vd->merge(newVD, vvVolDesc::VV_MERGE_SLABS2VOL);
    delete newVD;
    if (mergeErr != vvVolDesc::OK)
    {
      throw virvo::fileio::exception("format error: cannot merge slices");
    }
  }
}

}

namespace virvo {

namespace gdcm {

bool can_load(const vvVolDesc *vd)
{
  ::gdcm::Reader reader;
  reader.SetFileName(vd->getFilename());
  if(reader.CanRead())
    return true;

  return false;
}



dicom_meta load(vvVolDesc *vd)
{
  dicom_meta meta;
  bool subfile = false;

  namespace gdcm = ::gdcm;

  //const char *filename = argv[1];
  //std::cout << "filename: " << filename << std::endl;
  gdcm::Reader reader;
  reader.SetFileName(vd->getFilename());
  if( !reader.Read() )
  {
    throw fileio::exception("read error");
  }

  const gdcm::File &file = reader.GetFile();
  const gdcm::DataSet &ds = file.GetDataSet();
  gdcm::MediaStorage ms;
  ms.SetFromFile(file);
  /*
   * Until gdcm::MediaStorage is fixed only *compile* time constant will be handled
   * see -> http://chuckhahm.com/Ischem/Zurich/XX_0134
   * which make gdcm::UIDs useless :(
   */
  if( ms.IsUndefined() )
  {
    throw fileio::exception("unknown media storage");
  }

  gdcm::UIDs uid;
  uid.SetFromUID( ms.GetString() );
  if (!subfile)
    std::cerr << "MediaStorage is " << ms << " [" << uid.GetName() << "]" << std::endl;
  const gdcm::TransferSyntax &ts = file.GetHeader().GetDataSetTransferSyntax();
  uid.SetFromUID( ts.GetString() );
  if (!subfile)
    std::cerr << "TransferSyntax is " << ts << " [" << uid.GetName() <<  "]" << std::endl;

  if ( ms == gdcm::MediaStorage::MediaStorageDirectoryStorage )
  {
    load_dicom_dir(vd, reader, meta);
  }
  else if( gdcm::MediaStorage::IsImage( ms ) )
  {
      load_dicom_image(vd, meta);
    // Make big endian data:
    // TODO if (prop.littleEndian) vd->toggleEndianness(vd->frames-1);

    // Shift bits so that most significant used bit is leftmost:
    //vd->bitShiftData(pf.GetHighBit() - (pf.GetBitsAllocated() - 1), vd->frames-1);

    /*   if( md5sum )
         {
         char *buffer = new char[ image.GetBufferLength() ];
         image.GetBuffer( buffer );
         char digest[33] = {};
         gdcm::MD5::Compute( buffer, image.GetBufferLength(), digest );
         std::cerr << "md5sum: " << digest << std::endl;
         delete[] buffer;
         }*/
  }
  else if ( ms == gdcm::MediaStorage::EncapsulatedPDFStorage )
  {
    std::cerr << "  Encapsulated PDF File not supported yet" << std::endl;
    throw fileio::exception("format error: encapsulated PDF not supported");
  }
  // Do the IOD verification !
  //bool v = defs.Verify( file );
  //std::cerr << "IOD Verification: " << (v ? "succeed" : "failed") << std::endl;

  return meta;
}

} // namespace gdcm
} // namespace virvo

#endif
