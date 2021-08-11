Files layout
============
<SDK>/
	|
	|_drivers/
	|		galcore.ko 			# kernel HAL driver
	|		libGAL.so 			# user HAL driver
	|		libEGL.so 			# EGL driver
	|		libGLESv1_CM.so 		# OpenGL ES 1.1 driver
	|		libOpenVG.so		# OpenVG 1.1 driver
	|		libGLESv2.so		# OpenGL ES 2.0 driver
	|		libGLSLC.so		# OpenGL ES 2.0 shading language compiler
	|		libVDK.so			# Vivante Development Kit (USE_VDK=1)
	|		libVivante.so		# Pack all drivers except kernel driver in one library
	|	
	+---inc
	    |   gc_vdk_types.h
	    |   gc_vdk.h
        |
        +---HAL
        |       gc_hal.h
        |       gc_hal_types.h
        |       gc_hal_enum.h
        |       gc_hal_dump.h
        |       gc_hal_base.h
        |       gc_hal_raster.h
        |       gc_hal_options.h
        |       gc_hal_engine.h
        |       gc_hal_profiler.h
	    |
	    +---KHR
	    |       khrplatform.h
        |       khrvivante.h
	    |
	    +---EGL
	    |       egl.h
	    |       eglext.h
	    |       eglplatform.h
	    |       eglvivante.h
	    |
	    +---GLES
	    |       gl.h
	    |       glext.h
	    |       egl.h
	    |       glplatform.h
	    |       glrename.h
	    |
	    +---GLES2
	    |       gl2.h
	    |       gl2ext.h
	    |       gl2platform.h
	    |       gl2rename.h
	    |
	    \---VG
	            openvg.h
	            vgu.h
	            vgrename.h
                vgext.h
                vgplatform.h

Running applications on the target machine
==========================================

1. Copy the libraries to the target
	On the target machine:
	cp galcore.ko /
	cp libEGL.so libGLESv1_CM.so libGAL.so libGLSLC.so libGLESv2.so /lib

2. Install the kernel driver 
	insmod /galcore.ko registerMemBase=<REG_MEM_BASE> irqLine=<IRQ> contiguousSize=<CONTIGUOUS_MEM_SIZE>

	eg. On ARM EB development board:
	insmod /galcore.ko registerMemBase=0x80000000 irqLine=104 contiguousSize=0x400000

3. Run the application
	eg.
	cd $SDK_DIR/samples/vdk; ./tutorial1


