from nuscenes import NuScenes
from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.eval.detection.config import config_factory
import os

def main():
    # --- CONFIGURATION ---
    # 1. Path to your NuScenes data root (where v1.0-mini is)
    dataroot = 'data/nuscenes'
    
    # 2. Path to the JSON file you just generated
    # (Check if your file is named 'results_nusc.json' or 'results_nusc_results.json')
    result_path = 'save_output/pred_instances_3d/results_nusc.json' 
    
    # 3. Where to save the charts/tables
    output_dir = 'bevfusion_metrics_output'
    
    # --- LOAD GROUND TRUTH ---
    print(f"Loading NuScenes (v1.0-mini) from {dataroot}...")
    try:
        nusc = NuScenes(version='v1.0-mini', dataroot=dataroot, verbose=True)
    except Exception as e:
        print(f"Error loading NuScenes: {e}")
        print("Check if 'data/nuscenes/v1.0-mini' exists!")
        return

    # --- SETUP EVALUATOR ---
    print("Configuring evaluator...")
    cfg = config_factory('detection_cvpr_2019') # Standard config
    
    # Verify results file exists
    if not os.path.exists(result_path):
        print(f"ERROR: Could not find results file at: {result_path}")
        return

    nusc_eval = NuScenesEval(
        nusc, 
        config=cfg, 
        result_path=result_path, 
        eval_set='mini_val',  # <--- CRITICAL: Forces validation against Mini set
        output_dir=output_dir, 
        verbose=True
    )

    # --- RUN EVALUATION ---
    print("Running evaluation (this calculates mAP, NDS, etc)...")
    metrics_summary = nusc_eval.main(plot_examples=True, render_curves=True)
    metrics = metrics_summary

    print("\n" + "="*40)
    print("       FINAL RESULTS       ")
    print("="*40)
    print(f"mAP: {metrics['mean_ap']:.4f}")
    print(f"NDS: {metrics['nd_score']:.4f}")
    print("="*40)
    print(f"Full report and charts saved to: {output_dir}/")

if __name__ == '__main__':
    main()
