#! /bin/sh

# Expected variables:
# $1 build directory
# $2 src directory
# $3 verbose [ 0 | 1 ]

if [ $3 = 0 ];
then
  echo "Executing $0"
  echo "Build directory: $1"
  echo "Src directory: $2"
fi

FILENAME=$2/vvversioninfo.h
FILENAME_TMP=vvversioninfo.h~
PWD=`pwd`
COMMENT="// Auto generated file created using: $PWD/$0"
INFO=`(svn info $1 2> /dev/null | grep ^Revision: ) || ( git svn info 2> /dev/null | grep ^Revision: )`
REVISION=`echo $INFO | grep ^Revision: | sed -e 's/^Revision: /r/' || echo '(unknown)'`

if [ $3 = 0 ];
then
  echo "Current svn revision is $REVISION"
fi

echo $COMMENT > $FILENAME_TMP
echo "#ifndef VV_VERSION_INFO_H" >> $FILENAME_TMP
echo "#define VV_VERSION_INFO_H" >> $FILENAME_TMP
echo "#define VV_SVN_REVISION \"$REVISION\"" >> $FILENAME_TMP
echo "#define VV_VERSION \"2\"                            // major version change" >> $FILENAME_TMP
echo "#define VV_RELEASE \"01b\"                          // release counter" >> $FILENAME_TMP
echo "#define VV_YEAR 2005                              // year of release" >> $FILENAME_TMP
echo "#endif" >> $FILENAME_TMP

FILE_EXISTS=1
if [ -f "$FILENAME" ];
then
  FILE_EXISTS=0
else
  FILE_EXISTS=1
fi

REGENERATE=0
if [ $FILE_EXISTS = 0 ];
then
  DIFF=`/usr/bin/diff $FILENAME $FILENAME_TMP`;
  if [ -z "$DIFF" ];
  then
    if [ $3 = 0 ];
    then
      echo "No need to update $FILENAME";
    fi
    REGENERATE=1;
  else
    if [ $3 = 0 ];
    then
      echo "$FILENAME is out of date. Regenerate it.";
    fi
    REGENERATE=0;
  fi;
else
  if [ $3 = 0 ];
  then
    echo "$FILENAME doesn't exist. Generate it.";
  fi
fi

if [ $REGENERATE = 0 ];
then
  if [ $FILE_EXISTS = 0 ];
  then
    rm $FILENAME;
  fi
  mv $FILENAME_TMP $FILENAME;
else
  rm $FILENAME_TMP
fi;
