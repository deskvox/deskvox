# Microsoft Developer Studio Project File - Name="libvirvo" - Package Owner=<4>
# Microsoft Developer Studio Generated Build File, Format Version 6.00
# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) Static Library" 0x0104

CFG=libvirvo - Win32 Debug
!MESSAGE This is not a valid makefile. To build this project using NMAKE,
!MESSAGE use the Export Makefile command and run
!MESSAGE 
!MESSAGE NMAKE /f "libvirvo.mak".
!MESSAGE 
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "libvirvo.mak" CFG="libvirvo - Win32 Debug"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "libvirvo - Win32 Release" (based on "Win32 (x86) Static Library")
!MESSAGE "libvirvo - Win32 Debug" (based on "Win32 (x86) Static Library")
!MESSAGE 

# Begin Project
# PROP AllowPerConfigDependencies 0
# PROP Scc_ProjName ""
# PROP Scc_LocalPath ""
CPP=cl.exe
RSC=rc.exe

!IF  "$(CFG)" == "libvirvo - Win32 Release"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "Release"
# PROP BASE Intermediate_Dir "Release"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "Release"
# PROP Intermediate_Dir "Release"
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_MBCS" /D "_LIB" /YX /FD /c
# ADD CPP /nologo /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_MBCS" /D "_LIB" /YX /FD /c
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LIB32=link.exe -lib
# ADD BASE LIB32 /nologo
# ADD LIB32 /nologo

!ELSEIF  "$(CFG)" == "libvirvo - Win32 Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "Debug"
# PROP BASE Intermediate_Dir "Debug"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "Debug"
# PROP Intermediate_Dir "Debug"
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_MBCS" /D "_LIB" /YX /FD /GZ /c
# ADD CPP /nologo /W3 /Gm /GR /GX /Zi /Od /I ".." /I "..\include" /D "WIN32" /D "_DEBUG" /D "_MBCS" /D "_LIB" /D "HAVE_CG" /FD /GZ /c
# SUBTRACT CPP /YX
# ADD BASE RSC /l 0x409 /d "_DEBUG"
# ADD RSC /l 0x409 /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LIB32=link.exe -lib
# ADD BASE LIB32 /nologo
# ADD LIB32 /nologo /out:"..\obj\libvirvo.lib"

!ENDIF 

# Begin Target

# Name "libvirvo - Win32 Release"
# Name "libvirvo - Win32 Debug"
# Begin Group "Source Files"

# PROP Default_Filter "cpp;c;cxx;rc;def;r;odl;idl;hpj;bat"
# Begin Source File

SOURCE=..\vvdebugmsg.cpp
# End Source File
# Begin Source File

SOURCE=..\vvdicom.cpp
# End Source File
# Begin Source File

SOURCE=..\vvdynlib.cpp
# End Source File
# Begin Source File

SOURCE=..\vvfileio.cpp
# End Source File
# Begin Source File

SOURCE=..\vvgltools.cpp
# End Source File
# Begin Source File

SOURCE=..\vvimage.cpp
# End Source File
# Begin Source File

SOURCE=..\vvprintgl.cpp
# End Source File
# Begin Source File

SOURCE=..\vvrenderer.cpp
# End Source File
# Begin Source File

SOURCE=..\vvsocket.cpp
# End Source File
# Begin Source File

SOURCE=..\vvsocketio.cpp
# End Source File
# Begin Source File

SOURCE=..\vvsphere.cpp
# End Source File
# Begin Source File

SOURCE=..\vvstingray.cpp
# End Source File
# Begin Source File

SOURCE=..\vvstopwatch.cpp
# End Source File
# Begin Source File

SOURCE=..\vvtexrend.cpp
# End Source File
# Begin Source File

SOURCE=..\vvtfwidget.cpp
# End Source File
# Begin Source File

SOURCE=..\vvtokenizer.cpp
# End Source File
# Begin Source File

SOURCE=..\vvtoolshed.cpp
# End Source File
# Begin Source File

SOURCE=..\vvtransfunc.cpp
# End Source File
# Begin Source File

SOURCE=..\vvvecmath.cpp
# End Source File
# Begin Source File

SOURCE=..\vvvoldesc.cpp
# End Source File
# End Group
# Begin Group "Header Files"

# PROP Default_Filter "h;hpp;hxx;hm;inl"
# Begin Source File

SOURCE=..\vvarray.h
# End Source File
# Begin Source File

SOURCE=..\vvdebugmsg.h
# End Source File
# Begin Source File

SOURCE=..\vvdicom.h
# End Source File
# Begin Source File

SOURCE=..\vvdynlib.h
# End Source File
# Begin Source File

SOURCE=..\vvexport.h
# End Source File
# Begin Source File

SOURCE=..\vvfileio.h
# End Source File
# Begin Source File

SOURCE=..\vvglext.h
# End Source File
# Begin Source File

SOURCE=..\vvgltools.h
# End Source File
# Begin Source File

SOURCE=..\vvimage.h
# End Source File
# Begin Source File

SOURCE=..\vvprintgl.h
# End Source File
# Begin Source File

SOURCE=..\vvrenderer.h
# End Source File
# Begin Source File

SOURCE=..\vvsllist.h
# End Source File
# Begin Source File

SOURCE=..\vvsocket.h
# End Source File
# Begin Source File

SOURCE=..\vvsocketio.h
# End Source File
# Begin Source File

SOURCE=..\vvsphere.h
# End Source File
# Begin Source File

SOURCE=..\vvstingray.h
# End Source File
# Begin Source File

SOURCE=..\vvstopwatch.h
# End Source File
# Begin Source File

SOURCE=..\vvtexrend.h
# End Source File
# Begin Source File

SOURCE=..\vvtfwidget.h
# End Source File
# Begin Source File

SOURCE=..\vvtokenizer.h
# End Source File
# Begin Source File

SOURCE=..\vvtoolshed.h
# End Source File
# Begin Source File

SOURCE=..\vvtransfunc.h
# End Source File
# Begin Source File

SOURCE=..\vvvecmath.h
# End Source File
# Begin Source File

SOURCE=..\vvvirvo.h
# End Source File
# Begin Source File

SOURCE=..\vvvoldesc.h
# End Source File
# End Group
# End Target
# End Project
