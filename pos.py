"""
VERSION 20 FEB 2020
Compatible with python3 - Alex Pietrow
"""

import math
import numpy
import sys
import scipy.ndimage
import re
where,asfarray,asarray,array,zeros,arange = numpy.where,numpy.asfarray,numpy.asarray,numpy.array,numpy.zeros,numpy.arange


def create_circular_mask(h, w, center=None, radius=None):
    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])
    
    Y, X = numpy.ogrid[:h, :w]
    dist_from_center = numpy.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def aper2(img,x,y,r):
    #very quick aper knockoff
    h,w = img.shape
    mask = create_circular_mask(h, w, center=[x,y], radius=r)
    return numpy.sum(img*mask)


def regextract(filename, comments=False):
	"""
    Converts ds9 region files to become usable by the aper function.

    INPUTS:
        filename --  input ds9 regions file array.
        The ds9 file must be saved in physical coordinates. In DS9:
            Region->Save Regions
                [Choose destination/filename.reg and press OK]
            Format=ds9
            Coordinate System=physical
                [OK]
        

    OPTIONAL INPUTS:
        comments -- if comments=True then all circles must have comments. (Default = False)

    OUTPUTS:
        The output is an array of strings containing the values as shown below. This is done to enable the use of string names in comments.
        Even when comments are turned off, the format is kept to keep the format consistent.

        The format is 3xn if comments=False and 4xn if comments=True

        Array -- ['x','y','radius','comment']

    EXAMPLE:
        Convert the following region file into python format

            reg.ds9 contains:
            
            ================
            # Region file format: DS9 version 4.1
            global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
            physical
            circle(2763.4747,3175.7129,29.0882) # text={1}
            circle(2860.7076,3094.7166,25.0868) # text={text}
            ================
            
        Then calling:
            regions = regextract('reg.ds9', comment=True)
            
            regions then gives:
            array([['2763.4747','3175.7129','29.0882', '1'],
                 ['860.7076','3094.7166','25.0868', 'text'],],
                dtype='|S32')

            If the array does not contain text, setting it to be a float array is done by simply saying
            array.dtype = float

    REVISION HISTORY:
        Created by A.G.M. Pietrow 	                            22 Apr 2015
        Changed to give an array of floats - O. Burggraaff		6 May 2015
        Updated to be compatible with np 1.17 - AGM Pietrow     10 Jan 2020
        Updated to work with Python 3        - D.J.M Petit      20 Feb 2020

	"""
	array = numpy.array([])
	array2 = numpy.array([])
	regions = numpy.genfromtxt(filename, skip_header=3, comments='@', delimiter='\n' ,dtype='str')
	#print(regions)
	for line in regions: #for line in regions.split("\n"):
		array = numpy.append(array, numpy.array([str(x) for x in re.findall(r"\d+(?:\.\d+)?(?=[^()\n]*\))", line)]))
		
		if comments == True:		
			array2 = numpy.array([str(x) for x in re.findall(r"(?<=\{)[^}]+(?=\})", line)])

		array = numpy.append(array, array2)

	if comments == True:
		array = array.reshape(len(array)/4,4)
		x = array[:,0].astype(numpy.float)
		y = array[:,1].astype(numpy.float)
		r = array[:,2].astype(numpy.float)
		comments = array[:,3]
		return x,y,r,comments
	else:
		array = array.reshape(int(len(array)/3),3).astype(numpy.float)
		x = array[:,0]
		y = array[:,1]
		r = array[:,2]
		return x,y,r
	



def meanclip(image,mean,sigma,clipsig=3.,maxiter=5.,converge_num=0.02,verbose=False):

    """

 NAME:
       MEANCLIP

 PURPOSE:
       Computes an iteratively sigma-clipped mean on a data set
 EXPLANATION:
       Clipping is done about median, but mean is returned.
 CATEGORY:
       Statistics

 CALLING SEQUENCE:
       [mean,sigma]=MEANCLIP( image,mean,sigma, SUBS=
              CLIPSIG=, MAXITER=, CONVERGE_NUM=, VERBOSE=False, DOUBLE=False )

 INPUT POSITIONAL PARAMETERS:
       image:     Input data, any numeric array
       
 OUTPUT POSITIONAL PARAMETERS:
       Mean:     N-sigma clipped mean.
       Sigma:    Standard deviation of remaining pixels.

 INPUT KEYWORD PARAMETERS:
       CLIPSIG=3:  Number of sigma at which to clip.  Default=3
       MAXITER=5:  Ceiling on number of clipping iterations.  Default=5
       CONVERGE_NUM=0.02:  If the proportion of rejected pixels is less
           than this fraction, the iterations stop.  Default=0.02, i.e.,
           iteration stops if fewer than 2% of pixels excluded.
       VERBOSE=False:  Set this flag to get messages.
       DOUBLE=False - if set then perform all computations in double precision.
                 Otherwise double precision is used only if the input
                 data is double
 OUTPUT KEYWORD PARAMETER:
       SUBS:     Subscript array for pixels finally used. [not functional]


 MODIFICATION HISTORY:
       Written by:     RSH, RITSS, 21 Oct 98
       20 Jan 99 - Added SUBS, fixed misplaced paren on float call, 
                   improved doc.  RSH
       Nov 2005   Added /DOUBLE keyword, check if all pixels are removed  
                  by clipping W. Landsman 
       Nov 2012   converted to python G.P.P.L. Otten
       Feb 2015   removed by reference last=ct G.P.P.L. Otten
    """
    
    image=numpy.ravel(image)
    imagenumbers=numpy.arange(numpy.size(image))
    subs=imagenumbers[numpy.isfinite(image)]
    ct=numpy.sum(numpy.isfinite(image))
    iternr=0
    for iternr2 in numpy.arange(maxiter+1):
    #while iternr <= maxiter:
        skpix=image[subs]
        #print numpy.sum(skpix)
        #print numpy.size(skpix)
        iternr=iternr+1
        lastct=ct*1.
        medval=numpy.median(skpix)
        mean=numpy.mean(skpix,dtype=numpy.float64)
        sig=numpy.std(skpix,ddof=1,dtype=numpy.float64)
        wsm = (abs(skpix-medval) < clipsig*sig)
        ct=numpy.sum(wsm)
        if ct > 0:
            subs=subs[wsm]
        #print iternr
        if (iternr > maxiter) | (ct == 0) | (((abs(ct-lastct))/lastct) <= converge_num):
            break
    skpix=image[subs]
    mean=numpy.mean(skpix,dtype=numpy.float64)
    sig=numpy.std(skpix,ddof=1,dtype=numpy.float64)
    return [mean,sig]

def aper(image,xc,yc, phpadu, apr, setskyval, zeropoint=25,
         skyrad=[40,50], badpix=[0,0], minsky=[],
         skyalgorithm='sigmaclipping', exact = True, readnoise = 0,
         verbose=True, debug=False):
    """ Compute concentric aperture photometry on one ore more stars
    (adapted for IDL from DAOPHOT, then translated from IDL to Python).

    APER can compute photometry in several user-specified aperture radii.
    A separate sky value is computed for each source using specified inner
    and outer sky radii.

    By default, APER uses a magnitude system where a magnitude of
    25 corresponds to 1 flux unit. APER returns both
    fluxes and magnitudes.

     REQUIRED INPUTS:
         image  -  input image array
         xc     - scalar x value or 1D array of x coordinates.
         yc     - scalar y value or 1D array of y coordinates
         phpadu - Photons per Analog Digital Units, numeric scalar.  Converts
                   the data numbers in IMAGE to photon units.  (APER assumes
                   Poisson statistics.)
         apr    - scalar or 1D array of photometry aperture radii in pixel units.
      setskyval - Use this keyword to force the sky to a specified value
                   rather than have APER compute a sky value.    SETSKYVAL
                   can either be a scalar specifying the sky value to use for
                   all sources, or a 3 element vector specifying the sky value,
                   the sigma of the sky value, and the number of elements used
                   to compute a sky value.   The 3 element form of SETSKYVAL
                   is needed for accurate error budgeting.

     OPTIONAL KEYWORD INPUTS:

         zeropoint - zero point for converting flux (in ADU) to magnitudes
         skyrad - Two element list giving the inner and outer radii
                   to be used for the sky annulus
         badpix - Two element list giving the minimum and maximum value
                   of a good pix. If BADPIX[0] is equal to BADPIX[1] then
                   it is assumed that there are no bad pixels.

         exact -  By default, APER counts subpixels, but uses a polygon
                 approximation for the intersection of a circular aperture with
                 a square pixel (and normalize the total area of the sum of the
                 pixels to exactly match the circular area).   If the /EXACT
                 keyword, then the intersection of the circular aperture with a
                 square pixel is computed exactly.    The /EXACT keyword is much
                 slower and is only needed when small (~2 pixels) apertures are
                 used with very undersampled data.

         print - if set and non-zero then APER will also write its results to
                   a file aper.prt.   One can specify the output file name by
                   setting PRINT = 'filename'.
         verbose -  Print warnings, status, and ancillary info to the terminal
         skyalgorithm - set the algorithm by which the sky value is determined
                  Valid options are 'sigmaclipping' or 'mmm'.

     RETURNS:

         flux    -  NAPER by NSTAR array giving fluxes
         fluxerr -  NAPER by NSTAR array giving error in each flux

     PROCEDURES USED:
           MMM, PIXWT()
     NOTES:
           Reasons that a valid magnitude cannot be computed include the following:
          (1) Star position is too close (within 0.5 pixels) to edge of the frame
          (2) Less than 20 valid pixels available for computing sky
          (3) Modal value of sky could not be computed by the procedure MMM
          (4) *Any* pixel within the aperture radius is a "bad" pixel

           APER was modified in June 2000 in two ways: (1) the /EXACT keyword was
           added (2) the approximation of the intersection of a circular aperture
           with square pixels was improved (i.e. when /EXACT is not used)
     REVISON HISTORY:
           Adapted to IDL from DAOPHOT June, 1989                               B. Pfarr, STX
           Adapted for IDL Version 2,                                           J. Isensee, July, 1990
           Code, documentation spiffed up                                       W. Landsman   August 1991
           TEXTOUT may be a string                                              W. Landsman September 1995
           FLUX keyword added                                                   J. E. Hollis, February, 1996
           SETSKYVAL keyword, increase maxsky                                   W. Landsman, May 1997
           Work for more than 32767 stars                                       W. Landsman, August 1997
           Converted to IDL V5.0                                                W. Landsman   September 1997
           Don't abort for insufficient sky pixels                              W. Landsman  May 2000
           Added /EXACT keyword                                                 W. Landsman  June 2000
           Allow SETSKYVAL = 0                                                  W. Landsman  December 2000
           Set BADPIX[0] = BADPIX[1] to ignore bad pixels                       W. L.  January 2001
           Fix chk_badpixel problem introduced Jan 01 C. Ishida/                W.L. February 2001
           Converted from IDL to python                                         D. Jones January 2014
           Adapted for hstphot project                                          S. Rodney  July 2014
           
           taken from
           https://github.com/djones1040/PythonPhot/tree/master/PythonPhot
           Adapted for Obs1 based on Aper translation by GPPL Otten April 2013  AGM Pietrow Jan 2020
    """

    if verbose:
        import time
        tstart = time.time()
    elif verbose: import time

    if debug :
        import pdb
        pdb.set_trace()

    # Force np arrays
    if not numpy.iterable( xc ):
        xc = numpy.array([xc])
        yc = numpy.array([yc])
    if isinstance(xc, list): xc = numpy.array(xc)
    if isinstance(yc, list): xc = numpy.array(yc)
    assert len(xc) == len(yc), 'xc and yc arrays must be identical length.'

    if not numpy.iterable( apr ) :
        apr = numpy.array( [ apr ] )
    Naper = len( apr ) # Number of apertures
    Nstars = len( xc )   # Number of stars to measure

    # Set parameter limits
    if len(minsky) == 0: minsky = 20

    # Number of columns and rows in image array
    s = numpy.shape(image)
    ncol = s[1]
    nrow = s[0]


    if setskyval is not None :
        if not numpy.iterable(setskyval) :
            setskyval = [setskyval,0.,1.]
        assert len(setskyval)==3, 'Keyword SETSKYVAL must contain 1 or 3 elements'
        skyrad = [ 0., numpy.max(apr) + 1]    #use numpy.max (max function does not work on scalars)
    skyrad = numpy.asfarray(skyrad)




    # String array to display mags for all apertures in one line for each star
    outstr = [ '' for star in range(Nstars)]

    # Declare arrays
    mag = zeros( [ Nstars, Naper])
    magerr =  zeros( [ Nstars, Naper])
    flux = zeros( [ Nstars, Naper])
    fluxerr =  zeros( [ Nstars, Naper])
    badflag = zeros( [ Nstars, Naper])
    sky = zeros( Nstars )
    skyerr = zeros( Nstars )
    area = numpy.pi*apr*apr  # Area of each aperture

    if exact:
        bigrad = apr + 0.5
        smallrad = apr/numpy.sqrt(2) - 0.5

    if setskyval is None :
        rinsq =  skyrad[0]**2
        routsq = skyrad[1]**2

    #  Compute the limits of the submatrix.   Do all stars in vector notation.
    lx = (xc-skyrad[1]).astype(int)  # Lower limit X direction
    ly = (yc-skyrad[1]).astype(int)  # Lower limit Y direction
    ux = (xc+skyrad[1]).astype(int)  # Upper limit X direction
    uy = (yc+skyrad[1]).astype(int)  # Upper limit Y direction

    lx[where(lx < 0)[0]] = 0
    ux[where(ux > ncol-1)[0]] = ncol-1
    nx = ux-lx+1                         # Number of pixels X direction

    ly[where(ly < 0)[0]] = 0
    uy[where(uy > nrow-1)[0]] = nrow-1
    ny = uy-ly +1                      # Number of pixels Y direction

    dx = xc-lx                         # X coordinate of star's centroid in subarray
    dy = yc-ly                         # Y coordinate of star's centroid in subarray

    # Find the edge of the subarray that is closest to each star
    # and then flag any stars that are too close to the edge or off-image
    edge = zeros(len(dx))
    for i,dx1,nx1,dy1,ny1 in zip(range(len(dx)),dx,nx,dy,ny):
        edge[i] = min([(dx[i]-0.5),(nx[i]+0.5-dx[i]),(dy[i]-0.5),(ny[i]+0.5-dy[i])])
    badstar = numpy.where( (xc<0.5) | (xc>ncol-1.5) |
                        (yc<0.5) | (yc>nrow-1.5), 1, 0 )
    if numpy.any( badstar ) :
        nbad = badstar.sum()
        print('WARNING [aper.py] - ' + str(nbad) + ' star positions outside image')

    if verbose :
        tloop = time.time()
    for i in range(Nstars):  # Compute magnitudes for each star
        while True :
            # mimic GOTO statements : break out of this while block whenever
            # we decide this star is bad
            apflux = asarray([numpy.nan]*Naper)
            apfluxerr = asarray([numpy.nan]*Naper)
            apmag = asarray([numpy.nan]*Naper)
            apmagerr = asarray([numpy.nan]*Naper)
            skymod = 0.  # Sky mode
            skysig = 0.  # Sky sigma
            skyskw = 0.  # Sky skew
            error1 = asarray([numpy.nan]*Naper)
            error2 = asarray([numpy.nan]*Naper)
            error3 = array([numpy.nan]*Naper)
            apbad = numpy.ones( Naper )
            if badstar[i]: # star is bad, return NaNs for all values
                break

            rotbuf = image[ ly[i]:uy[i]+1,lx[i]:ux[i]+1 ] #Extract subarray from image
            shapey,shapex = numpy.shape(rotbuf)[0],numpy.shape(rotbuf)[1]
            #  RSQ will be an array, the same size as ROTBUF containing the square of
            #      the distance of each pixel to the center pixel.

            dxsq = ( arange( nx[i] ) - dx[i] )**2
            rsq = numpy.ones( [ny[i], nx[i]] )
            for ii  in range(ny[i]):
                rsq[ii,:] = dxsq + (ii-dy[i])**2

            if exact:
                nbox = range(nx[i]*ny[i])
                xx = (nbox % nx[i]).reshape( ny[i], nx[i])
                yy = (nbox/nx[i]).reshape(ny[i],nx[i])
                x1 = numpy.abs(xx-dx[i])
                y1 = numpy.abs(yy-dy[i])
            else:
                r = numpy.sqrt(rsq) - 0.5    #2-d array of the radius of each pixel in the subarray

            rsq,rotbuf = rsq.reshape(shapey*shapex),rotbuf.reshape(shapey*shapex)
            if setskyval is None :
                # skypix will be 1-d array of sky pixels
                skypix = numpy.zeros( rsq.shape )

                #  Select pixels within sky annulus,
                skypix[where(( rsq >= rinsq ) &
                             ( rsq <= routsq ))[0]] = 1
                if badpix[0]!=badpix[1] :
                    # Eliminate pixels above or below the badpix threshold vals
                    skypix[where(((rotbuf < badpix[0]) | (rotbuf > badpix[1])) &
                                 (skypix == 1))[0]] = 0
                sindex =  where(skypix)[0]
                nsky = len(sindex)

                if ( nsky < minsky ):   # Insufficient sky pixels?
                    if verbose:
                        print("ERROR: nsky=%i is fewer than minimum %i valid pixels in the sky annulus."%(nsky,minsky))
                    break

                skybuf = rotbuf[ sindex[0:nsky] ]
                if skyalgorithm.startswith('sigmaclip'):
                    # The sky annulus is (nearly) empty of stars, (as in a diff image)
                    # so we can simply compute the sigma-clipped mean of all pixels in
                    # the annulus
                    skybufclipped,lothresh,hithresh = sigmaclip( skybuf, low=4.0, high=4.0)
                    skymod = numpy.mean( skybufclipped )
                    skysig = numpy.std( skybufclipped )
                    skyskw = skew( skybufclipped )

                else:
                    # Compute the sky mode, sigma and skewness using the
                    # mean/median/mode algorithm in mmm.py, which assumes that
                    # most of the outlier pixels are positive.
                    skymod, skysig, skyskw = mmm.mmm(skybuf,readnoise=readnoise,minsky=minsky)

                skyvar = skysig**2    #Variance of the sky brightness
                sigsq = skyvar/nsky  #Square of standard error of mean sky brightness
             
                if ( skysig < 0.0 ):
                    # If the modal sky value could not be determined, then all
                    # apertures for this star are bad.  So skip to the next.
                    break

                if skysig > 999.99: skysig = 999      #Don't overload output formats
                if skyskw < -99: skyskw = -99
                if skyskw > 999.9: skyskw = 999.9

            else:
                skymod = setskyval[0]
                skysig = setskyval[1]
                nsky = setskyval[2]
                skyvar = skysig**2
                sigsq = skyvar/nsky
                skyskw = 0

            for k in range(Naper): # Find pixels within each aperture
                if ( edge[i] >= apr[k] ):   #Does aperture extend outside the image?
                    if exact:
                        mask = zeros(ny[i]*nx[i])

                        x1,y1 = x1.reshape(ny[i]*nx[i]),y1.reshape(ny[i]*nx[i])
                        igoodmag = where( ( x1 < smallrad[k] ) & (y1 < smallrad[k] ))[-1]
                        Ngoodmag = len(igoodmag)
                        if Ngoodmag > 0: mask[igoodmag] = 1
                        bad = where(  (x1 > bigrad[k]) | (y1 > bigrad[k] ))[-1]
                        mask[bad] = -1

                        gfract = where(mask == 0.0)[0]
                        Nfract = len(gfract)
                        if Nfract > 0:
                            yygfract = yy.reshape(ny[i]*nx[i])[gfract]
                            xxgfract = xx.reshape(ny[i]*nx[i])[gfract]

                            mask[gfract] = Pixwt(dx[i],dy[i],apr[k],xxgfract,yygfract)
                            mask[gfract[where(mask[gfract] < 0.0)[0]]] = 0.0
                        thisap = where(mask > 0.0)[0]

                        thisapd = rotbuf[thisap]
                        fractn = mask[thisap]
                    else:
                        # approximating the circular aperture shape
                        rshapey,rshapex = numpy.shape(r)[0],numpy.shape(r)[1]
                        thisap = where( r.reshape(rshapey*rshapex) < apr[k] )[0]   # Select pixels within radius
                        thisapd = rotbuf.reshape(rshapey*rshapex)[thisap]
                        thisapr = r.reshape(rshapey*rshapex)[thisap]
                        fractn = apr[k]-thisapr
                        fractn[where(fractn > 1)[0]] = 1
                        fractn[where(fractn < 0)[0]] = 0  # Fraction of pixels to count
                        full = zeros(len(fractn))
                        full[where(fractn == 1)[0]] = 1.0
                        gfull = where(full)[0]
                        Nfull = len(gfull)
                        gfract = where(1 - full)[0]
                        factor = (area[k] - Nfull ) / numpy.sum(fractn[gfract])
                        fractn[gfract] = fractn[gfract]*factor
                else:
                    if verbose :
                        print("WARNING [aper.py]: aperture extends outside the image!")
                    continue
                    # END "if exact ...  else ..."

                # Check for any bad pixel values (nan,inf) and those outside
                # the user-specified range of valid pixel values.  If any
                # are found in the aperture, raise the badflux flag.
                apbad[k] = 0
                if not numpy.all( numpy.isfinite(thisapd) ) :
                    if verbose :
                        print("WARNING : nan or inf pixels detected in aperture.\n"
                              "We're setting these to 0, but the photometry"
                              "may be biased.")
                    thisapd[numpy.isfinite(thisapd)==False] = 0
                    apbad[k] = 1
                    fractn = 0
                if badpix[0] < badpix[1] :
                    ibadpix = numpy.where((thisapd<=badpix[0]) | (thisapd>=badpix[1]))
                    if len(ibadpix[0]) > 0 :
                        if verbose :
                            print("WARNING : pixel values detected in aperture"
                                  " that are outside of the allowed range "
                                  " [%.1f , %.1f] \n"%(badpix[0],badpix[1]) +
                                  "We're treating these as 0, but the "
                                  "photometry may be biased.")
                        thisapd[ibadpix] = 0
                        apbad[k] = 1
                # Sum the flux over the irregular aperture
                apflux[k] = numpy.sum(thisapd*fractn)
            # END for loop over apertures

            igoodflux = where(numpy.isfinite(apflux))[0]
            Ngoodflux = len(igoodflux)
            if Ngoodflux > 0:
                if verbose > 2 :
                    print(" SRCFLUX   APFLUX    SKYMOD   AREA")
                    for igf in igoodflux :
                        print("%.4f   %.4f   %.4f   %.4f "%(apflux[igf]-skymod*area[igf],apflux[igf],skymod,area[igf]))
                # Subtract sky from the integrated brightnesses
                apflux[igoodflux] = apflux[igoodflux] - skymod*area[igoodflux]

            # Compute flux error
            error1[igoodflux] = area[igoodflux]*skyvar   #Scatter in sky values
            error2[igoodflux] = numpy.abs(apflux[igoodflux])/phpadu  #Random photon noise
            error3[igoodflux] = sigsq*area[igoodflux]**2  #Uncertainty in mean sky brightness
            apfluxerr[igoodflux] = numpy.sqrt(error1[igoodflux] + error2[igoodflux] + error3[igoodflux])

            igoodmag = where (apflux > 0.0)[0]  # Are there any valid integrated fluxes?
            Ngoodmag = len(igoodmag)
            if ( Ngoodmag > 0 ) : # convert valid fluxes to mags
                apmagerr[igoodmag] = 1.0857*apfluxerr[igoodmag]/apflux[igoodmag]   #1.0857 = 2.5/log(10)
                apmag[igoodmag] =  zeropoint-2.5*numpy.log10(apflux[igoodmag])
            break # Closing the 'while True' loop.

        # TODO : make a more informative output string
        outstr[i] = '%.3f,%.3f :'%(xc[i],yc[i]) + \
                    '  '.join( [ '%.4f+-%.4f'%(apmag[ii],apmagerr[ii])
                                 for ii in range(Naper) ] )

        sky[i] = skymod
        skyerr[i] = skysig
        mag[i,:] = apmag
        magerr[i,:]= apmagerr
        flux[i,:] = apflux
        fluxerr[i,:]= apfluxerr
        badflag[i,:] = apbad

    if Nstars == 1 :
        sky = sky[0]
        skyerr = skyerr[0]
        mag = mag[0]
        magerr = magerr[0]
        flux = flux[0]
        fluxerr = fluxerr[0]
        badflag = badflag[0]
        outstr = outstr[0]

    if verbose:
        print('hstphot.aper took %.3f seconds'%(time.time()-tstart))
        print('Each of %i loops took %.3f seconds'%(Nstars,(time.time()-tloop)/Nstars))

    return(flux[0],fluxerr[0])

def mmm(sky_vector,skymod,sigma,skew,highbad=[],debug=False,readnoise=[],maxiter=50.,minsky=20.,integer=False,silent=False):
    #print sky_vector
    sky_vector=numpy.ravel(sky_vector)
    sky = numpy.sort(sky_vector)
    Nsky=numpy.size(sky_vector)
    Nlast=int(Nsky-1.)
    if Nsky < minsky:
        sigma=-1.0
        skew = 0.0
        print('ERROR -Input vector must contain at least the minimal amount of elements')
        return [skymod,sigma,skew]
    skymid=numpy.median(sky)
    cut1=numpy.min(numpy.array([skymid-numpy.min(sky),numpy.max(sky)-skymid]))
    if numpy.size(highbad) == 1:
        cut1=numpy.min(numpy.array( [cut1,highbad - skymid]))
            
    cut2=skymid+cut1
    cut1=skymid-cut1

    good = ( (sky <= cut2) & (sky >= cut1))
    Ngood=numpy.sum(good)
    good=(numpy.arange(numpy.size(sky)))[good]
    delta = sky[good] - skymid
    tot = numpy.sum(delta,dtype='float64')                     
    totsq = numpy.sum(delta**2,dtype='float64')
    if ( Ngood == 0 ):
        sigma=-1.0
        skew = 0.0   
        print('ERROR - No sky values fall within cuts')
        return [skymod,sigma,skew]
    
    minimm=int(numpy.min(good)-1)
    maximm=int(numpy.max(good))
    
    skymed = numpy.median(sky[good])
    skymn = tot/(maximm-minimm)
    sigma = numpy.std(sky[good])
    skymn = skymn + skymid
    
    if (skymed < skymn):
        skymod = (3.*skymed)-(2.*skymn)
    else:
        skymod=skymn*1.
    
    
    clamp=1.
    old=0.
    niter=0
    #redo=True
    for niter1 in numpy.arange(maxiter+1):
    #while(redo == 1):
        niter=niter+1
        if niter > maxiter:
            sigma=-1.
            skew=0.
            print('Too many iterations')
            return [skymod,sigma,skew]
        if maximm-minimm < minsky:
            sigma=-1.
            skew=0.
            print('Too few valid sky elements')
            return [skymod,sigma,skew]
        

        r=numpy.log10(maximm-minimm)
        r=numpy.max(numpy.array([2.,( -0.1042*r + 1.1695)*r + 0.8895 ]))
        cut=r*sigma+0.5*numpy.abs(skymn-skymod)
        if integer == True:
            cut=numpy.max(numpy.array([cut,1.5]))
        cut1=skymod-cut
        cut2=skymod+cut
    
        redo=False
        newmin=int(minimm*1.)
        tst_min=sky[newmin+1] >= cut1
        done = (newmin == -1) & (tst_min)
        if done == False:
            done = (sky[numpy.max(numpy.array([newmin,0.]))] < cut1) & (tst_min)
        if done == False:
            
            if tst_min == True:
                istep = -1
            else:
                istep=1
            
            for niter2 in numpy.arange(Nsky):
            #while(done == False):
                newmin=newmin+istep
                done= (newmin == Nlast) | (newmin == -1)
                if done == False:
                    done = (sky[newmin] <= cut1) & (sky[newmin+1] >= cut1)
                if done == True:
                    break
            if tst_min == True:
                delta = sky[(newmin+1):(minimm+1)] - skymid
            else:
                delta= sky[(minimm+1):(newmin+1)] - skymid
            tot=tot-istep*numpy.sum(delta,dtype="float64")
            totsq=totsq-istep*numpy.sum(delta**2,dtype="float64")
            redo=True
            minimm = int(newmin*1.)
            
        newmax=int(maximm*1.)
        tst_max = (sky[maximm] <= cut2)
        done = (maximm == Nlast) & tst_max
        if done == False:
            done=tst_max & (sky[numpy.min(numpy.array([maximm+1,Nlast]))] > cut2)
        if done == False:
            if tst_max == False:
                istep = -1
            else:
                istep=1
            
            for niter3 in numpy.arange(Nsky):
                newmax=newmax+istep
                done= (newmax == Nlast) | (newmax == -1)
                if done == False:
                    done = (sky[newmax] <= cut2) & (sky[newmax+1] >= cut2)
                if done == True:
                    break
            if tst_max == True:
                delta=sky[(maximm+1):(newmax+1)]-skymid
            else:
                delta=sky[(newmax+1):(maximm+1)]-skymid
            tot=tot+istep*numpy.sum(delta,dtype="float64")
            totsq=totsq+istep*numpy.sum(delta**2,dtype="float64")
            redo=True
            maximm=int(newmax*1.)
            

        Nsky = maximm - minimm
        if ( Nsky < minsky ):
            sigma = -1.0
            skew = 0.0
            print('ERROR - Outlier rejection left too few sky elements')
            return [skymod,sigma,skew]
               
            
        skymn = tot/Nsky
        sigma = numpy.sqrt( numpy.max( numpy.array([(totsq/Nsky - skymn**2),0.]) ))
        skymn = skymn + skymid
        
        
        CENTER = (minimm + 1 + maximm)/2.
        SIDE = int(numpy.round(0.2*(maximm-minimm)))/2.  + 0.25
        j = int(numpy.round(CENTER-SIDE))
        k = int(numpy.round(CENTER+SIDE))
        
        if numpy.size(readnoise) > 0:
            L = int(numpy.round(CENTER-0.25))
            M = int(numpy.round(CENTER+0.25))
            R = 0.25*readnoise
            while ((j > 0) & (k < Nsky-1) & ( ((sky[L] - sky[j]) < R) | ((sky[k] - sky[M]) < R))):
                j=j-1
                k=k+1
        
        skymed = numpy.sum(sky[j:(k+1)],dtype="float64")/(k-j+1)
        if (skymed < skymn):
            dmod = 3.*skymed-2.*skymn-skymod
        else:
            dmod = skymn-skymod
        if dmod*old < 0.:
            clamp = 0.5*clamp
        skymod=skymod+clamp*dmod
        old=dmod*1.
        if redo == False:
            break

    skew=(skymn-skymod)/numpy.max([1.,sigma])
    Nsky=maximm-minimm
    if (debug==True):
        print('% MMM: Number of unrejected sky elements: ',Nsky)
        print('% MMM: Number of iterations: ',niter)
        print('% MMM: Mode, Sigma, Skew of sky vector:', skymod, sigma, skew  ) 
 
                
    return [skymod,sigma,skew]

where,abs,sqrt,greater,less_equal,less,greater_equal = \
    numpy.where,numpy.abs,numpy.sqrt,numpy.greater,numpy.less_equal,numpy.less,numpy.greater_equal

def Pixwt(xc, yc, r, x, y):
    """Circle-rectangle overlap area computation.

    Compute the fraction of a unit pixel that is interior to a circle.
    The circle has a radius r and is centered at (xc, yc).  The center of
    the unit pixel (length of sides = 1) is at (x, y).
    
    area = pixwt.Pixwt( xc, yc, r, x, y )

    INPUT PARAMETERS:
         xc, yc - Center of the circle, numeric scalars
    	 r      - Radius of the circle, numeric scalars
    	 x, y   - Center of the unit pixel, numeric scalar or vector

    RETURNS:
         Function value: Computed overlap area.

    EXAMPLE:
         What is the area of overlap of a circle with radius 3.44 units centered
         on the point 3.23, 4.22 with the pixel centered at [5,7]
    
         import pixwt
         pixwt.Pixwt(3.23,4.22,3.44,5,7)  ==>  0.6502

    PROCEDURE:
         Divides the circle and rectangle into a series of sectors and
    	 triangles.  Determines which of nine possible cases for the
    	 overlap applies and sums the areas of the corresponding sectors
    	 and triangles.  Called by aper.pro
    
    NOTES:
         If improved speed is needed then a C version of this routines, with
         notes on how to linkimage it to IDL is available at
         ftp://ftp.lowell.edu/pub/buie/idl/custom/
    
    MODIFICATION HISTORY:
         Ported by Doug Loucks, Lowell Observatory, 1992 Sep, from the
         routine pixwt.c, by Marc Buie.
         converted to Python by D. Jones
    """
    return Intarea( xc, yc, r, x-0.5, x+0.5, y-0.5, y+0.5 )

def Arc( x, y0, y1, r):
    """; Function Arc( x, y0, y1, r )
    ;
    ; Compute the area within an arc of a circle.  The arc is defined by
    ; the two points (x,y0) and (x,y1) in the following manner:  The circle
    ; is of radius r and is positioned at the origin.  The origin and each
    ; individual point define a line which intersects the circle at some
    ; point.  The angle between these two points on the circle measured
    ; from y0 to y1 defines the sides of a wedge of the circle.  The area
    ; returned is the area of this wedge.  If the area is traversed clockwise
    ; then the area is negative, otherwise it is positive.
    ; ---------------------------------------------------------------------------"""
    return 0.5 * r*r * ( numpy.arctan( (y1).astype(float)/(x).astype(float) ) - numpy.arctan( (y0).astype(float)/(x).astype(float) ) )

def Chord( x, y0, y1):
    """; ---------------------------------------------------------------------------
    ; Function Chord( x, y0, y1 )
    ;
    ; Compute the area of a triangle defined by the origin and two points,
    ; (x,y0) and (x,y1).  This is a signed area.  If y1 > y0 then the area
    ; will be positive, otherwise it will be negative.
    ; ---------------------------------------------------------------------------"""
    return 0.5 * x * ( y1 - y0 )

def Oneside( x, y0, y1, r):
    """; ---------------------------------------------------------------------------
    ; Function Oneside( x, y0, y1, r )
    ;
    ; Compute the area of intersection between a triangle and a circle.
    ; The circle is centered at the origin and has a radius of r.  The
    ; triangle has verticies at the origin and at (x,y0) and (x,y1).
    ; This is a signed area.  The path is traversed from y0 to y1.  If
    ; this path takes you clockwise the area will be negative.
    ; ---------------------------------------------------------------------------"""

    true = 1
    size_x  = numpy.shape( x )
    if not size_x: size_x = [0]

    if size_x[ 0 ] == 0:
      if x == 0: return x
      elif abs( x ) >= r: return Arc( x, y0, y1, r )
      yh = sqrt( r*r - x*x )
      if ( y0 <= -yh ):
          if ( y1 <= -yh ) : return Arc( x, y0, y1, r )
          elif ( y1 <=  yh ) : return Arc( x, y0, -yh, r ) \
                  + Chord( x, -yh, y1 )
          else          : return Arc( x, y0, -yh, r ) \
                  + Chord( x, -yh, yh ) + Arc( x, yh, y1, r )
      
      elif ( y0 <  yh ):
          if ( y1 <= -yh ) : return Chord( x, y0, -yh ) \
                  + Arc( x, -yh, y1, r )
          elif ( y1 <=  yh ) : return Chord( x, y0, y1 )
          else : return Chord( x, y0, yh ) + Arc( x, yh, y1, r )

      else          :
          if ( y1 <= -yh ) : return Arc( x, y0, yh, r ) \
                               + Chord( x, yh, -yh ) + Arc( x, -yh, y1, r )
          elif ( y1 <=  yh ) : return Arc( x, y0, yh, r ) + Chord( x, yh, y1 )
          else          : return Arc( x, y0, y1, r )

    else :
        ans2 = x
        t0 = where( x == 0)[0]
        count = len(t0)
        if count == len( x ): return ans2

        ans = x * 0
        yh = x * 0
        to = where( abs( x ) >= r)[0]
        tocount = len(to)
        ti = where( abs( x ) < r)[0]
        ticount = len(ti)
        if tocount != 0: ans[ to ] = Arc( x[to], y0[to], y1[to], r )
        if ticount == 0: return ans
        
        yh[ ti ] = sqrt( r*r - x[ti]*x[ti] )
        
        t1 = where( numpy.less_equal(y0[ti],-yh[ti]) )[0]
        count = len(t1)
        if count != 0:
            i = ti[ t1 ]

            t2 = where( numpy.less_equal(y1[i],-yh[i]))[0]
            count = len(t2)
            if count != 0:
                j = ti[ t1[ t2 ] ]
                ans[j] =  Arc( x[j], y0[j], y1[j], r )

            t2 = where( ( greater(y1[i],-yh[i]) ) &
                        ( less_equal(y1[i],yh[i]) ))[0]
            count = len(t2)
            if count != 0:
                j = ti[ t1[ t2 ] ]
                ans[j] = Arc( x[j], y0[j], -yh[j], r ) \
                    + Chord( x[j], -yh[j], y1[j] )

            t2 = where( greater(y1[i], yh[i]) )[0]
            count = len(t2)

            if count != 0:
                j = ti[ t1[ t2 ] ]
                ans[j] = Arc( x[j], y0[j], -yh[j], r ) \
                    + Chord( x[j], -yh[j], yh[j] ) \
                    + Arc( x[j], yh[j], y1[j], r )

        t1 = where( ( greater(y0[ti],-yh[ti]) ) &
                    ( less(y0[ti],yh[ti]) ))[0]
        count = len(t1)
        if count != 0:
            i = ti[ t1 ]

            t2 = where( numpy.less_equal(y1[i],-yh[i]))[0]
            count = len(t2)
            if count != 0:
                j = ti[ t1[ t2 ] ]
                ans[j] = Chord( x[j], y0[j], -yh[j] ) \
                    + Arc( x[j], -yh[j], y1[j], r )


            t2 = where( ( greater(y1[i], -yh[i]) ) &
                        ( less_equal(y1[i], yh[i]) ))[0]
            count = len(t2)

            if count != 0:
                j = ti[ t1[ t2 ] ]
                ans[j] = Chord( x[j], y0[j], y1[j] )

            t2 = where( greater(y1[i], yh[i]))[0]
            count = len(t2)
            if count != 0:
                j = ti[ t1[ t2 ] ]
                ans[j] = Chord( x[j], y0[j], yh[j] ) \
                    + Arc( x[j], yh[j], y1[j], r )

        t1 = where( greater_equal(y0[ti], yh[ti]))[0]
        count = len(t1)
        if count != 0:
            i = ti[ t1 ]

            t2 = where ( numpy.less_equal(y1[i], -yh[i]))[0]
            count = len(t2)
            if count != 0:
                j = ti[ t1[ t2 ] ]
                ans[j] = Arc( x[j], y0[j], yh[j], r ) \
                    + Chord( x[j], yh[j], -yh[j] ) \
                    + Arc( x[j], -yh[j], y1[j], r )

            t2 = where( ( greater(y1[i], -yh[i]) ) &
                        ( less_equal(y1[i], yh[i]) ))[0]
            count = len(t2)
            if count != 0:
                j = ti[ t1[ t2 ] ]
                ans[j] = Arc( x[j], y0[j], yh[j], r ) \
                    + Chord( x[j], yh[j], y1[j] )

            t2 = where( greater(y1[i], yh[i]))[0]
            count = len(t2)
            if count != 0:
                j = ti[ t1[ t2 ] ]
                ans[j] = Arc( x[j], y0[j], y1[j], r )

        return ans

def Intarea( xc, yc, r, x0, x1, y0, y1):
    """; ---------------------------------------------------------------------------
    ; Function Intarea( xc, yc, r, x0, x1, y0, y1 )
    ;
    ; Compute the area of overlap of a circle and a rectangle.
    ;    xc, yc  :  Center of the circle.
    ;    r       :  Radius of the circle.
    ;    x0, y0  :  Corner of the rectangle.
    ;    x1, y1  :  Opposite corner of the rectangle.
    ; ---------------------------------------------------------------------------"""

#
# Shift the objects so that the circle is at the origin.
#
    x0 = x0 - xc
    y0 = y0 - yc
    x1 = x1 - xc
    y1 = y1 - yc

    return Oneside( x1, y0, y1, r ) + Oneside( y1, -x1, -x0, r ) +\
        Oneside( -x0, -y1, -y0, r ) + Oneside( -y0, x0, x1, r )
   

