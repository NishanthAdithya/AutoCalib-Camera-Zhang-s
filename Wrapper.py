import numpy as np
import cv2
from pathlib import Path
from calib import detect_cb_cor, generate_world_pts, estimate_homo, extract_K_from_homo, extract_Rt, optimize_calib, reproj_error, undistort_im



def autocalib(im_dir = "Calibration_Imgs",patt_size = (6, 9),sq_size = 21.5, vis= False):
    
    # get all the image files
    im_dir_path = Path(im_dir)
    im_files = sorted([f for f in im_dir_path.glob("*.jpg")])
    

    # get world points
    world_pts = generate_world_pts(patt_size, sq_size)
    # detect corners in all images
    all_im_pts = []
    all_world_pts = []
    valid_images = []
    
    for i, im_path in enumerate(im_files):
        corners = detect_cb_cor(str(im_path), patt_size)
        
        if corners is not None:
            all_im_pts.append(corners)
            all_world_pts.append(world_pts)
            valid_images.append(im_path)
    
   
    homo = []
    for i, (im_pts, world_pts) in enumerate(zip(all_im_pts, all_world_pts)):
        H = estimate_homo(im_pts, world_pts)
        homo.append(H)
        
    K_init = extract_K_from_homo(homo)
  
    
    all_R_init = []
    all_t_init = []
    for i, H in enumerate(homo):
        R, t = extract_Rt(H, K_init)
        all_R_init.append(R)
        all_t_init.append(t)
     
    # init distortion parameters (assumed zero)
    k_init = np.array([0.0, 0.0])
   
    
    # compute init reprojection error
    init_error = reproj_error( K_init, k_init, all_world_pts, all_im_pts, all_R_init, all_t_init)
    
    
    K_opt, k_opt, all_R_opt, all_t_opt = optimize_calib( K_init, k_init, all_world_pts, all_im_pts, all_R_init, all_t_init)
    
    
    # Compute final reprojection error
    final_error = reproj_error(K_opt, k_opt, all_world_pts,all_im_pts, all_R_opt, all_t_opt)
    
   
    save_calib_im(valid_images, all_im_pts, all_world_pts,
                          K_opt, k_opt, all_R_opt, all_t_opt)
    
    # Additional visualization if requested
    if vis:
        vis_results(valid_images, all_im_pts, all_world_pts,
                          K_opt, k_opt, all_R_opt, all_t_opt)
    
    return {'K': K_opt,'k': k_opt,'R_list': all_R_opt, 't_list': all_t_opt,'reprojection_error': final_error,'world_pts': all_world_pts,'im_pts': all_im_pts}


def save_calib_im(im_paths, all_im_pts, all_world_pts,K, k_dist, all_R, all_t):

    from calib import project_pts
    
    
    output_dir = Path("Results")
    output_dir.mkdir(exist_ok=True)
    
    for i, im_path in enumerate(im_paths):
        im = cv2.imread(str(im_path))
        if im is None:
            continue
        
        im_name = im_path.stem
        
        # createe image with detected and reprojected corners
        im_with_corners = im.copy()
        
        # plot detected corners
        for pt in all_im_pts[i]:
            cv2.circle(im_with_corners, tuple(pt.astype(int)), 8, (0, 255, 0), 2)
        
        # project world points and draw reprojected corners 
        projected = project_pts(all_world_pts[i], K, all_R[i], all_t[i], k_dist)
        for pt in projected:
            cv2.circle(im_with_corners, tuple(pt.astype(int)), 5, (0, 0, 255), 2)
        
        
        num_pts = len(all_im_pts[i])
        if num_pts == 54:
            r, c = 6, 9
            
            corners_2d = all_im_pts[i].reshape(r, c, 2)
                
            for r in range(r):
                for c in range(c-1):
                    pt1 = tuple(corners_2d[r, c].astype(int))
                    pt2 = tuple(corners_2d[r, c+1].astype(int))
                    cv2.line(im_with_corners, pt1, pt2, (0, 255, 0), 1)
            for c in range(c):
                for r in range(r-1):
                    pt1 = tuple(corners_2d[r, c].astype(int))
                    pt2 = tuple(corners_2d[r+1, c].astype(int))
                    cv2.line(im_with_corners, pt1, pt2, (0, 255, 0), 1)
        
        # rectify image
        im_undistorted = undistort_im(im, K, k_dist)
        
        # save images
        corners_path = output_dir / f"{im_name}_with_cor.jpg"
        cv2.imwrite(str(corners_path), im_with_corners)
       
        
        # rectified_path = output_dir / f"{im_name}_rect.jpg"
        # cv2.imwrite(str(rectified_path), im_undistorted)

    


def vis_results(im_paths, all_im_pts, all_world_pts, K, k_dist, all_R, all_t):
   
    from calib import project_pts
    
    
    output_dir = Path("Results")
    output_dir.mkdir(exist_ok=True)
    
    
    
    for i, im_path in enumerate(im_paths):
        im = cv2.imread(str(im_path))
        if im is None:
            continue
        
        im_name = im_path.stem
        
        # create side-by-side comparison
        im_org = im.copy()
        im_undistorted = undistort_im(im, K, k_dist)
        
        # draw corners on both
        for pt in all_im_pts[i]:
            cv2.circle(im_org, tuple(pt.astype(int)), 5, (0, 255, 0), 2)
        
        projected = project_pts(all_world_pts[i], K, all_R[i], all_t[i], k_dist)
        for pt in projected:
            cv2.circle(im_undistorted, tuple(pt.astype(int)), 5, (0, 0, 255), 2)
        
        #create side-by-side image
        h, w = im.shape[:2]
        comparison = np.hstack([im_org, im_undistorted])
        
        # add labels
        # cv2.putText(comparison, "Org", (10, 30), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # cv2.putText(comparison, "Rect", (w + 10, 30), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # comparison_path = output_dir / f"{im_name}_comp.jpg"
        # cv2.imwrite(str(comparison_path), comparison)


def main():

    results = autocalib(im_dir='Calibration_Imgs')
    print("K = ")
    print(results['K'])
    print("k1, k2 = ")
    print(results['k'])
    print("reprojection error= ")
    print(results['reprojection_error'])

if __name__ == "__main__":
    exit(main())
