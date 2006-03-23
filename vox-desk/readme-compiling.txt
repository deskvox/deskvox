Deskvox developers notes
========================

- Sourceforge: cvs update:
  cvs -d:ext:jpschulze@cvs.sourceforge.net:/cvsroot/deskvox up
  or set CVSROOT to :ext:jpschulze@cvs.sourceforge.net:/cvsroot/deskvox

- Compiler flags:
  > #define HAVE_CG to compile with support for pixel shaders
  > #define USE_STINGRAY to compile with support for Peter Stephenson's Stingray renderer
  > #define HAVE_SOAP to compile with SOAP support in Windows; adds menu item to download data sets from server
  > #define SAGE_APP to integrate with EVL's SAGE API for tiled display support; it won't run standalone anymore then

- Problem: runtime error in dynamic_cast:
  Turn on Run-Time Type Information (RTTI) in Project Settings -> C/C++ -> C++ Language

- Problem: When starting an application, a pop-up appears with: 
           "The application failed to initialize properly (0xc0000022)" 
  Solution: In Cygwin, make sure that the DLLs the program needs have the executable bit set.

- If you need to know which libraries are linked by the linker, add /VERBOSE:LIB
  to Linker/Command Line/Additional Options

- In order to prevent multiply defined linker errors in vshell.dll add libcmtd.lib
  to Linker/Input/Ignore Specific Library

- If afxres.h cannot be found in a .rc file, replace all occurrences with winres.h.

- In Visual C++ Virvo projects should be created as Win32 Console 
  applications to display program information. Thus they do not need
  a WinMain() but just a main() method for startup.

- In Visual C++ make sure Projekt/Einstellungen/Linker Objekt-/Bibliothek-Module
  contains opengl32.lib and wsock32.lib

- In Visual C++ for all projects in Projekt/Einstellungen/Linker the
  "Name der Ausgabedatei" should be ..\bin\xxx
    
- Make sure the maximum warning level is enabled

- To compile NVidia Cg pixel shader program (not currently needed as pixel shaders are compiled at runtime):
  > right-click virvo_shader.arbfp1.cg -> Settings
  > Commands:
    "C:\Programs\NVIDIA-Cg\bin\cgc" -profile arbfp1 -o ../../virvo-1.0/obj/$(InputName).pp $(InputPath)
  > Outputs: 
    ..\..\virvo-1.0\obj\$(InputName).pp
    
- Runtime error: The application failed to initialize properly (0xc0000022). Click on OK to terminate the application.