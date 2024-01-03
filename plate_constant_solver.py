import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astroquery.gaia import Gaia



Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source" #GAIA catalog from 2016


#open directory folder
cal_path = "E:/stardata/MEE2024.00000971.Zenith-Center2.fit"
#r"C:\Users\kyleg\OneDrive\Desktop\MEE 2024\Testing\Testing_1_ref.fts"
Gaia.ROW_LIMIT = 10                                         #limit of number of rows that can be returned
with fits.open(cal_path) as hdul:                           #benefit of this: automatically closes fits file without clos()
    hdr = hdul[0].header                                    #Header information
    image_data = hdul[0].data                               #Data information
    print(str(hdr))
    print()
    print(repr(hdr))                                        #prints all info regarding the header
    print()
    '''
    indices_of_ref_star = int(hdr["NUMREF"]) - 1            #stores info on length of list 
    avg_date = (hdr["DATE-SRT"] + hdr["DATE-END"])/2        #averages the julian date of exposure
    alpha_0 = float(hdr['RA']) * 15 * (np.pi/180)           #Center of fits image RA in radians. 
    delta_0 = float(hdr['DEC']) * (np.pi/180)               #Center of fits image DEC in radians. 
    focal = hdr['FOCAL']                                    #focal length of telesope
    xpixel = hdr['PIXELX'] * 1e-6                           #xpixel length in meters
    ypixel = hdr['PIXELY'] * 1e-6                           #ypixel length in meters
    '''
    #indices_of_ref_star = int(hdr["NUMREF"]) - 1            #stores info on length of list 
    avg_date = hdr["DATE-OBS"]        #averages the julian date of exposure

    sc = SkyCoord(ra=hdr['OBJCTRA'], dec=hdr['OBJCTDEC'], unit=(u.hourangle, u.deg))
    print(sc)

    alpha_0 = sc.ra * (np.pi/180)           #Center of fits image RA in radians. 
    delta_0 = sc.dec * (np.pi/180)               #Center of fits image DEC in radians. 
    #focal = hdr['FOCAL']                                    #focal length of telesope
    xpixel = hdr['XPIXSZ'] * 1e-6                           #xpixel length in meters
    ypixel = hdr['XPIXSZ'] * 1e-6                           #ypixel length in meters

    #This creates a list of dictionaries containing the name of the reference stars, and their ra and dec from GAIA
    cele_name_list = []         
    gaia_source_ID_final_list = []  
    epoch_julian_year_final_list = []
    ra_from_epoch_final_list = []
    dec_from_epoch_final_list = []
    ra_error_final_list = []
    dec_error_final_list = []
    pmra_final_list = []
    pmra_error_final_list = []
    pmdec_final_list = []
    pmdec_error_final_list = []
    

    width = u.Quantity(0.1, u.deg)                         #width of GAIA query that the API will look at
    height = u.Quantity(0.1, u.deg)                        #length of GAIA query that the API will look at
    for i in range(indices_of_ref_star):                   #will iterate through each reference star
        name_keyword = 'STAR_' + str(i)                    #creating keyword so that data from fits can be accessed
        name_star = hdr[name_keyword]                      #re-stores the name of the star as variable
        cele_name_list.append(name_star)                   #stores name of star in list
        cele_coord = SkyCoord.from_name(name_star)         #gets coordinates of star
        r = Gaia.query_object_async(coordinate=cele_coord, width=width, height=height)  #results from GAIA query

        #These lists are needed to store the data for 2nd while loop, then when the second for-loop is exited, its previous data is wiped
        gaia_source_ID_list = []
        epoch_julian_year_list = []
        ra_from_epoch_list = []
        dec_from_epoch_list = []
        ra_error_list = []
        dec_error_list = []
        pmra_list = []
        pmra_error_list = []
        pmdec_list = []
        pmdec_error_list = []
        error_pred_ra_list = []
        error_dist_list = []
            
        #There are multiple resources on a single object, so it needs to get parsed through and have best one chosen based on least error
        for j in range(len(r)):
            gaia_source_ID = str(r[j]['source_id'])             #grabs source ID
            epoch = float(r[j]['ref_epoch'])                    #2016
            epoch_year = Time(epoch, format='decimalyear')      #converts 2016 to Julian year
            epoch_julian_year = float(epoch_year.jd)            #restores it as float
            ra_from_epoch = float(r[j]['ra'])                   #ra in degrees
            ra_error_from_epoch = str(r[j]['ra_error'])         #ra error
            dec_from_epoch = float(r[j]['dec'])                 #dec in degrees
            dec_error_from_epoch = str(r[j]['dec_error'])       #dec error
            pmra = str(r[j]['pmra'])                            #proper motion of ra in mas/year
            pmra_error = str(r[j]['pmra_error'])                #proper motion of ra error
            pmdec = str(r[j]['pmdec'])                          #proper motion of dec in mas/year
            pmdec_error = str(r[j]['pmdec_error'])              #proper motion of dec error
            Δt = avg_date - epoch_julian_year                   #time that has passed since epoch and time at which image was taken
            
            #if no error can be found, not a valid choice
            #len(string) ==2 was chosen because if data wasn't provided, it would be --
            #type of varaible is extremely important for this
            #all data being ran through if-statments were stored as strings so len() could be used as parameter
            if len(ra_error_from_epoch) == 2:
                ra_error_from_epoch = 'INVALID'
            else:
                ra_error_from_epoch = float(ra_error_from_epoch) #redefined as float for math purposes

            #if no error can be found, not a valid choice
            if len(dec_error_from_epoch) == 2:
                dec_error_from_epoch = 'INVALID'
            else:
                dec_error_from_epoch = float(dec_error_from_epoch)

            #if no movement is found, value is set to 0
            if len(pmra) == 2:
                pmra = 0
            else:
                pmra = float(pmra)
            
            #if no movement is found, value is set to 0
            if len(pmdec) == 2:
                pmdec = 0
            else:
                pmdec = float(pmdec)

            #if no error can be found, not a valid choice
            if len(pmra_error) == 2:
                pmra_error = 'INVALID'
            else:
                pmra_error = float(pmra_error)

            #if no error can be found, not a valid choice
            if len(pmdec_error) == 2:
                pmdec_error = 'INVALID'
            else:
                pmdec_error =float(pmdec_error)

            #If none of the error values are strings
            if type(ra_error_from_epoch) is float and type(pmra_error) is float and type(dec_error_from_epoch) is float and type(pmdec_error) is float:
                error_pred_ra = np.sqrt((ra_error_from_epoch**2) + ((Δt**2)*(pmra_error**2)))     #error propagation of ra
                error_pred_dec = np.sqrt((dec_error_from_epoch**2) + ((Δt**2)*(pmdec_error**2)))  #error propagation of dec
                error_dist = np.sqrt(error_pred_ra**2 + error_pred_dec**2)                        #sum of square of ra and dec
                error_dist_list.append(error_dist)                                                #storing data to list
            else:
                pass

            gaia_source_ID_list.append(gaia_source_ID)
            epoch_julian_year_list.append(epoch_julian_year)
            ra_from_epoch_list.append(ra_from_epoch)
            dec_from_epoch_list.append(dec_from_epoch)
            ra_error_list.append(ra_error_from_epoch)
            dec_error_list.append(dec_error_from_epoch)
            pmra_list.append(pmra)
            pmdec_list.append(pmdec)
            pmra_error_list.append(pmra_error)
            pmdec_error_list.append(pmdec_error)
            
        #This finds the lowest error that choses that GAIA info to get data from 
        error_min = min(error_dist_list)                            #finds smallest value                            
        error_min_loc = error_dist_list.index(error_min)            #finds index of that value

        gaia_source_ID_final_list.append(gaia_source_ID_list[error_min_loc]) 
        epoch_julian_year_final_list.append(epoch_julian_year_list[error_min_loc]) 
        ra_from_epoch_final_list.append(ra_from_epoch_list[error_min_loc]) 
        dec_from_epoch_final_list.append(dec_from_epoch_list[error_min_loc]) 
        ra_error_final_list.append(ra_error_list[error_min_loc]) 
        dec_error_final_list.append(dec_error_list[error_min_loc]) 
        pmra_final_list.append(pmra_list[error_min_loc]) 
        pmdec_final_list.append( pmdec_list[error_min_loc]) 
        pmra_error_final_list.append(pmra_error_list[error_min_loc]) 
        pmdec_error_final_list.append(pmdec_error_list[error_min_loc]) 


        print(f'Minimum error: {error_min} located at {error_min_loc}')
        print()

    print()
    print(f'Celestial object names: {cele_name_list}')
    print()
    print(f'GAIA Source ID: {gaia_source_ID_final_list}')
    print()
    print(f'Julian Date of epoch: {epoch_julian_year_final_list}') 
    print()
    print(f'Right Ascension at epoch in degrees: {ra_from_epoch_final_list}')
    print()
    print(f'Declination of epoch in degrees: {dec_from_epoch_final_list}')
    print()
    print(f'Right Ascension error: {ra_error_final_list}')
    print()
    print(f'Declination error: {dec_error_final_list}')
    print()
    print(f'Proper motion of RA [mas/year]: {pmra_final_list}')
    print()
    print(f'Error in proper motion of RA [mas/year]: {pmra_error_final_list}')
    print()
    print(f'Proper motion of DEC [mas/year]: {pmdec_final_list}')
    print()
    print(f'Error in proper motion of DEC [mas/year]: {pmdec_error_final_list}')
    print()
    


    
        #alpha = cele_coord_list[j][name_star][0]
        #delta = cele_coord_list[j][name_star][1]
        #X = (np.cos(delta) * np.sin(alpha-alpha_0)) / ((np.cos(delta_0)*np.cos(delta)*np.cos(alpha - alpha_0)) + (np.sin(delta_0)*np.sin(delta)))
        #Y = ((np.sin(delta_0)*np.cos(delta)*np.cos(alpha-alpha_0)) - (np.cos(delta_0)*np.sin(delta))) / ((np.cos(delta_0)*np.cos(delta)*np.cos(alpha - alpha_0)) + (np.sin(delta_0)*np.sin(delta)))
        
        #x = X*focal
        #y = Y*focal
        #print(x,y)
        #print()




#use for-loop to parse between fits file
#open first fits file
#Get meta-data from fits file(header)
#Biggest piece of meta-data needed: julian date, RA,DEC of center of image, width and height of pixels, focal length of telescope, 
#4 or more reference stars contained inside fits image
#Get RA and DEC coordinate of reference stars via catalog(GAIA)
#Transformation from (RA, DEC) -> (α,δ) -> (Χ,Υ) -> (x,y)
#Units of Coordinates:
#(RA,DEC)- Hours for RA and degrees in DEC
#(α,δ)- Degrees for α and δ
#(Χ,Υ)- standard plate coordinates in cartensian radians
#(x,y)- pixel coordinates in millimeters
#Use 4 or more (x,y) values to solve for  a,b,c,d,e,f constants
#Store constants, focal length, camera type, telescope type into text file for further use
