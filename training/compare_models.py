"""
Compare CNN Classifier Performance: With PSO vs Without PSO
Loads both models and evaluates them on the same test set
"""
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from utils.preprocess import normalize_landmarks

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    X = df.iloc[:, 1:].values.astype('float32')
    y = pd.factorize(df.iloc[:, 0])[0]
    labels = pd.factorize(df.iloc[:, 0])[1]  # Get label names
    X = normalize_landmarks(X)
    return train_test_split(X, y, test_size=0.2, random_state=42), labels

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model and return metrics"""
    predictions = model.predict(X_test, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    
    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test)
    
    # Calculate per-class accuracy
    correct_per_class = []
    for i in range(26):
        mask = y_test == i
        if np.sum(mask) > 0:
            class_acc = np.mean(y_pred[mask] == y_test[mask])
            correct_per_class.append(class_acc)
    
    avg_class_acc = np.mean(correct_per_class)
    
    return {
        'name': model_name,
        'accuracy': accuracy,
        'avg_class_accuracy': avg_class_acc,
        'predictions': y_pred,
        'confidences': np.max(predictions, axis=1)
    }

if __name__ == "__main__":
    print("=" * 80)
    print("MODEL COMPARISON: CNN with PSO vs CNN without PSO")
    print("=" * 80)
    
    # Load data
    print("\n[1/5] Loading test data...")
    (X_train, X_test, y_train, y_test), labels = load_data('landmarks/augmented_landmarks.csv')
    print(f"✓ Test samples: {len(X_test)}")
    
    # Load encoder
    print("\n[2/5] Loading encoder...")
    encoder = tf.keras.models.load_model('models/encoder.h5', compile=False)
    X_test_enc = encoder.predict(X_test, verbose=0)
    X_test_enc = X_test_enc[..., np.newaxis]
    print(f"✓ Encoded test shape: {X_test_enc.shape}")
    
    # Load models
    print("\n[3/5] Loading trained classifiers...")
    try:
        model_pso = tf.keras.models.load_model('models/classifier.h5')
        print("✓ PSO-optimized model loaded")
    except:
        print("✗ PSO model not found (models/classifier.h5)")
        model_pso = None
    
    try:
        model_no_pso = tf.keras.models.load_model('models/classifier_no_pso.h5')
        print("✓ No-PSO model loaded")
    except:
        print("✗ No-PSO model not found (models/classifier_no_pso.h5)")
        model_no_pso = None
    
    if model_pso is None and model_no_pso is None:
        print("\n❌ Error: No models found! Train the models first.")
        exit(1)
    
    # Evaluate models
    print("\n[4/5] Evaluating models on test set...")
    print("-" * 80)
    
    results = []
    
    if model_pso:
        print("\nEvaluating PSO-optimized model...")
        pso_results = evaluate_model(model_pso, X_test_enc, y_test, "CNN + PSO")
        results.append(pso_results)
        print(f"✓ PSO Model Accuracy: {pso_results['accuracy']*100:.2f}%")
    
    if model_no_pso:
        print("\nEvaluating default (no PSO) model...")
        no_pso_results = evaluate_model(model_no_pso, X_test_enc, y_test, "CNN (Default)")
        results.append(no_pso_results)
        print(f"✓ No-PSO Model Accuracy: {no_pso_results['accuracy']*100:.2f}%")
    
    # Display comparison
    print("\n[5/5] Comparison Results")
    print("=" * 80)
    print("\n📊 ACCURACY COMPARISON TABLE")
    print("┌" + "─" * 78 + "┐")
    print("│ Model                │ Test Accuracy │ Avg Class Acc │ Improvement      │")
    print("├" + "─" * 78 + "┤")
    
    if len(results) == 2:
        # Both models available
        baseline = no_pso_results
        optimized = pso_results
        
        improvement_abs = (optimized['accuracy'] - baseline['accuracy']) * 100
        improvement_rel = ((optimized['accuracy'] / baseline['accuracy']) - 1) * 100
        
        print(f"│ CNN (Default/No PSO) │    {baseline['accuracy']*100:6.2f}%    │    {baseline['avg_class_accuracy']*100:6.2f}%    │   Baseline       │")
        print(f"│ CNN + PSO Optimized  │    {optimized['accuracy']*100:6.2f}%    │    {optimized['avg_class_accuracy']*100:6.2f}%    │   +{improvement_abs:5.2f}%       │")
        print("└" + "─" * 78 + "┘")
        
        print("\n📈 PERFORMANCE METRICS")
        print("┌" + "─" * 78 + "┐")
        print(f"│ Absolute Improvement:  {improvement_abs:6.2f}% accuracy gain                            │")
        print(f"│ Relative Improvement:  {improvement_rel:6.2f}% better performance                        │")
        print(f"│ Error Reduction:       {((baseline['accuracy'] - optimized['accuracy']) / (1 - baseline['accuracy'])) * -100:6.2f}% fewer errors                          │")
        print("└" + "─" * 78 + "┘")
        
        # Confidence comparison
        avg_conf_baseline = np.mean(baseline['confidences'])
        avg_conf_optimized = np.mean(optimized['confidences'])
        
        print("\n🎯 PREDICTION CONFIDENCE")
        print("┌" + "─" * 78 + "┐")
        print(f"│ CNN (No PSO):    Avg Confidence = {avg_conf_baseline*100:5.2f}%                              │")
        print(f"│ CNN + PSO:       Avg Confidence = {avg_conf_optimized*100:5.2f}%                              │")
        print(f"│ Difference:                        {(avg_conf_optimized - avg_conf_baseline)*100:+5.2f}%                              │")
        print("└" + "─" * 78 + "┘")
        
        # Errors comparison
        errors_baseline = np.sum(baseline['predictions'] != y_test)
        errors_optimized = np.sum(optimized['predictions'] != y_test)
        
        print("\n❌ ERROR ANALYSIS")
        print("┌" + "─" * 78 + "┐")
        print(f"│ Total Test Samples:        {len(y_test):5d}                                       │")
        print(f"│ CNN (No PSO) Errors:       {errors_baseline:5d} mistakes                                  │")
        print(f"│ CNN + PSO Errors:          {errors_optimized:5d} mistakes                                  │")
        print(f"│ Errors Prevented by PSO:   {errors_baseline - errors_optimized:5d} fewer mistakes                         │")
        print("└" + "─" * 78 + "┘")
        
    else:
        # Only one model
        for result in results:
            print(f"│ {result['name']:20s} │    {result['accuracy']*100:6.2f}%    │    {result['avg_class_accuracy']*100:6.2f}%    │   N/A            │")
        print("└" + "─" * 78 + "┘")
    
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)
    
    # Save results to file
    with open('comparison_results.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MODEL COMPARISON RESULTS: CNN with PSO vs CNN without PSO\n")
        f.write("=" * 80 + "\n\n")
        
        if len(results) == 2:
            f.write(f"CNN (Default/No PSO):\n")
            f.write(f"  - Test Accuracy: {baseline['accuracy']*100:.4f}%\n")
            f.write(f"  - Errors: {errors_baseline}\n\n")
            
            f.write(f"CNN + PSO Optimized:\n")
            f.write(f"  - Test Accuracy: {optimized['accuracy']*100:.4f}%\n")
            f.write(f"  - Errors: {errors_optimized}\n\n")
            
            f.write(f"Improvement:\n")
            f.write(f"  - Absolute: +{improvement_abs:.4f}%\n")
            f.write(f"  - Relative: +{improvement_rel:.4f}%\n")
            f.write(f"  - Errors Prevented: {errors_baseline - errors_optimized}\n")
    
    print("\n💾 Results saved to: comparison_results.txt")
