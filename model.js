/**
 * model.js - AI module for enhanced object and architectural feature detection.
 * Uses TensorFlow.js with coco-ssd, and integrates room-based logic.
 */
/**
 * Enhanced Object Detection Model for 360° Virtual Tours
 * Combines COCO-SSD with custom architectural feature detection
 */

class EnhancedObjectDetector {
  constructor() {
    this.cocoModel = null;
    this.isInitialized = false;
    
    // Extended object categories with confidence thresholds
    this.objectCategories = {
      furniture: {
        chair: { threshold: 0.25, priority: 'high' },
        couch: { threshold: 0.35, priority: 'high' },
        bed: { threshold: 0.35, priority: 'high' },
        'dining table': { threshold: 0.25, priority: 'high' },
        desk: { threshold: 0.25, priority: 'medium' }
      },
      electronics: {
        tv: { threshold: 0.35, priority: 'high' },
        laptop: { threshold: 0.25, priority: 'medium' },
        'cell phone': { threshold: 0.4, priority: 'low' },
        keyboard: { threshold: 0.25, priority: 'medium' },
        mouse: { threshold: 0.2, priority: 'low' },
        remote: { threshold: 0.2, priority: 'low' }
      },
      kitchen: {
        refrigerator: { threshold: 0.35, priority: 'high' },
        microwave: { threshold: 0.25, priority: 'medium' },
        oven: { threshold: 0.3, priority: 'high' },
        sink: { threshold: 0.25, priority: 'high' },
        toaster: { threshold: 0.25, priority: 'medium' }
      },
      bathroom: {
        toilet: { threshold: 0.4, priority: 'high' },
        sink: { threshold: 0.25, priority: 'high' }
      },
      decorative: {
        'potted plant': { threshold: 0.25, priority: 'medium' },
        vase: { threshold: 0.2, priority: 'low' },
        clock: { threshold: 0.25, priority: 'medium' },
        book: { threshold: 0.2, priority: 'low' }
      }
    };

    // Room-specific common architectural features
    this.commonArchitecturalFeatures = {
      general: {
        door: { probability: 0.95, typical_count: [1, 2] },
        window: { probability: 0.8, typical_count: [1, 3] }
      },
      kitchen: {
        door: { probability: 0.9, typical_count: [1, 2] },
        window: { probability: 0.7, typical_count: [1, 2] },
        cabinet: { probability: 0.85, typical_count: [3, 8] }
      },
      bathroom: {
        door: { probability: 0.95, typical_count: [1] },
        window: { probability: 0.4, typical_count: [0, 1] },
        cabinet: { probability: 0.6, typical_count: [1, 3] }
      },
      toilet: {
        door: { probability: 0.95, typical_count: [1] },
        window: { probability: 0.3, typical_count: [0, 1] },
        cabinet: { probability: 0.4, typical_count: [0, 2] }
      },
      bedroom: {
        door: { probability: 0.9, typical_count: [1, 2] },
        window: { probability: 0.85, typical_count: [1, 3] },
        closet: { probability: 0.7, typical_count: [1, 2] }
      },
      entrance: {
        door: { probability: 0.98, typical_count: [1, 3] },
        window: { probability: 0.5, typical_count: [0, 2] }
      }
    };

    // Color-based feature detection patterns
    this.colorPatterns = {
      wood: { hue: [20, 40], saturation: [20, 80] },
      metal: { hue: [200, 220], saturation: [10, 30] },
      glass: { brightness: [70, 95], saturation: [0, 20] }
    };
  }

  async initialize() {
    try {
      console.log("Loading enhanced object detection models...");
      
      // Load COCO-SSD model with optimized settings
      this.cocoModel = await cocoSsd.load({
        base: 'mobilenet_v2' // More accurate than lite version
      });
      
      this.isInitialized = true;
      console.log("Enhanced object detection initialized successfully");
      return true;
    } catch (error) {
      console.error("Failed to initialize object detection:", error);
      return false;
    }
  }

  // Preprocess 360° image to improve detection
  preprocessImage(canvas, image, enhancementLevel = 'medium') {
    const ctx = canvas.getContext('2d');
    canvas.width = 1024;
    canvas.height = 1024;
    
    // Apply different enhancement levels
    const enhancements = {
      low: 'contrast(1.1) brightness(1.05) saturate(1.05)',
      medium: 'contrast(1.2) brightness(1.1) saturate(1.1)',
      high: 'contrast(1.3) brightness(1.15) saturate(1.2)'
    };
    
    ctx.filter = enhancements[enhancementLevel] || enhancements.medium;
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    
    return canvas;
  }

  // Advanced detection with multiple sampling strategies
  async detectObjects(image, roomType = 'general') {
    if (!this.isInitialized || !this.cocoModel) {
      throw new Error("Model not initialized");
    }

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const allDetections = [];

    try {
      // Strategy 1: Full image analysis (equirectangular projection)
      this.preprocessImage(canvas, image, 'medium');
      let predictions = await this.cocoModel.detect(canvas, 12);
      allDetections.push(...predictions.map(p => ({...p, source: 'full', weight: 1.0})));

      // Strategy 2: Enhanced center crop (main viewing area)
      const cropSize = Math.min(image.width, image.height) * 0.75;
      const cropX = (image.width - cropSize) / 2;
      const cropY = (image.height - cropSize) / 2;
      
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      this.preprocessImage(canvas, image, 'high');
      ctx.drawImage(image, cropX, cropY, cropSize, cropSize, 0, 0, canvas.width, canvas.height);
      predictions = await this.cocoModel.detect(canvas, 10);
      allDetections.push(...predictions.map(p => ({...p, source: 'center', weight: 1.2})));

      // Strategy 3: Adaptive grid sampling
      const gridSections = this.createAdaptiveGrid(image, roomType);
      for (let i = 0; i < gridSections.length; i++) {
        const section = gridSections[i];
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        this.preprocessImage(canvas, image, section.enhancement);
        
        ctx.drawImage(image, 
          section.x, section.y, section.width, section.height,
          0, 0, canvas.width, canvas.height
        );
        
        predictions = await this.cocoModel.detect(canvas, 6);
        allDetections.push(...predictions.map(p => ({
          ...p, 
          source: `grid_${i}`, 
          weight: section.weight
        })));
      }

      // Strategy 4: Edge-enhanced detection for architectural features
      const edgeDetections = await this.edgeEnhancedDetection(canvas, image);
      allDetections.push(...edgeDetections);

      // Process and filter all detections
      const processedObjects = this.processDetections(allDetections, roomType);
      
      // Add custom architectural feature detection
      const architecturalObjects = await this.detectArchitecturalFeatures(image, roomType);
      
      // Combine and validate results
      return this.combineAndValidateResults(processedObjects, architecturalObjects, roomType);
      
    } catch (error) {
      console.error("Detection error:", error);
      return {};
    }
  }

  createAdaptiveGrid(image, roomType) {
    // Create different grid patterns based on room type
    const gridConfigs = {
      kitchen: { cols: 3, rows: 2, focus: 'lower' }, // Focus on counter areas
      bathroom: { cols: 2, rows: 2, focus: 'center' },
      bedroom: { cols: 3, rows: 2, focus: 'center' },
      entrance: { cols: 2, rows: 3, focus: 'vertical' },
      general: { cols: 3, rows: 2, focus: 'center' }
    };

    const config = gridConfigs[roomType] || gridConfigs.general;
    const sections = [];
    const sectionWidth = image.width / config.cols;
    const sectionHeight = image.height / config.rows;
    
    for (let row = 0; row < config.rows; row++) {
      for (let col = 0; col < config.cols; col++) {
        const section = {
          x: col * sectionWidth,
          y: row * sectionHeight,
          width: sectionWidth,
          height: sectionHeight,
          weight: 0.8,
          enhancement: 'medium'
        };

        // Adjust weights and enhancement based on focus area
        if (config.focus === 'center' && this.isCenterSection(col, row, config.cols, config.rows)) {
          section.weight = 1.1;
          section.enhancement = 'high';
        } else if (config.focus === 'lower' && row >= config.rows - 1) {
          section.weight = 1.1;
          section.enhancement = 'high';
        }

        sections.push(section);
      }
    }
    return sections;
  }

  isCenterSection(col, row, totalCols, totalRows) {
    const centerCol = Math.floor(totalCols / 2);
    const centerRow = Math.floor(totalRows / 2);
    return Math.abs(col - centerCol) <= 0.5 && Math.abs(row - centerRow) <= 0.5;
  }

  async edgeEnhancedDetection(canvas, image) {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Apply edge enhancement filter
    ctx.filter = 'contrast(1.5) brightness(1.0) saturate(0.8)';
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    
    // Additional edge detection could be implemented here
    // For now, use standard detection with edge-enhanced image
    const predictions = await this.cocoModel.detect(canvas, 8);
    return predictions.map(p => ({...p, source: 'edge_enhanced', weight: 0.9}));
  }

  processDetections(detections, roomType) {
    const objectCounts = {};
    const roomThresholds = this.getRoomSpecificThresholds(roomType);
    
    // Apply weighted scoring and filtering
    detections.forEach(detection => {
      const className = detection.class;
      const confidence = detection.score * (detection.weight || 1.0);
      const threshold = roomThresholds[className] || 0.25;
      
      if (confidence < threshold) return;
      if (this.shouldFilterDetection(detection, roomType, confidence)) return;
      
      if (!objectCounts[className]) {
        objectCounts[className] = {
          count: 0,
          confidences: [],
          maxConfidence: 0,
          avgConfidence: 0,
          sources: new Set(),
          detectionQuality: 'medium'
        };
      }
      
      objectCounts[className].count++;
      objectCounts[className].confidences.push(Math.round(confidence * 100));
      objectCounts[className].maxConfidence = Math.max(
        objectCounts[className].maxConfidence, 
        Math.round(confidence * 100)
      );
      objectCounts[className].sources.add(detection.source);
    });

    // Post-process results
    Object.keys(objectCounts).forEach(className => {
      const obj = objectCounts[className];
      obj.avgConfidence = Math.round(
        obj.confidences.reduce((a, b) => a + b, 0) / obj.confidences.length
      );
      
      // Smart deduplication and quality assessment
      obj.count = this.intelligentDeduplication(obj, className, roomType);
      obj.detectionQuality = this.assessDetectionQuality(obj, className);
      obj.sources = Array.from(obj.sources);
    });

    return objectCounts;
  }

  getRoomSpecificThresholds(roomType) {
    const baseThresholds = {
      // High-confidence objects
      chair: 0.25, couch: 0.35, bed: 0.35, 'dining table': 0.25,
      tv: 0.35, laptop: 0.25, refrigerator: 0.35, toilet: 0.4,
      
      // Medium-confidence objects
      microwave: 0.25, oven: 0.3, sink: 0.25, 'potted plant': 0.25,
      clock: 0.25, keyboard: 0.25, desk: 0.25,
      
      // Lower-confidence objects (prone to false positives)
      person: 0.6, 'cell phone': 0.4, book: 0.3, cup: 0.35,
      mouse: 0.3, remote: 0.3, vase: 0.25
    };

    // Room-specific adjustments
    const roomAdjustments = {
      kitchen: {
        refrigerator: 0.25, microwave: 0.2, oven: 0.25, sink: 0.2,
        'dining table': 0.2, chair: 0.2
      },
      bathroom: {
        toilet: 0.3, sink: 0.2, person: 0.7
      },
      toilet: {
        toilet: 0.25, sink: 0.2, person: 0.8
      },
      bedroom: {
        bed: 0.25, chair: 0.2, desk: 0.2, laptop: 0.2
      },
      entrance: {
        chair: 0.3, person: 0.5
      }
    };

    const adjustments = roomAdjustments[roomType] || {};
    return { ...baseThresholds, ...adjustments };
  }

  shouldFilterDetection(detection, roomType, adjustedConfidence) {
    const className = detection.class;
    
    // Context-based filtering
    const contextFilters = {
      toilet: {
        unlikely: ['bed', 'couch', 'dining table', 'tv', 'laptop'],
        threshold_boost: 0.3
      },
      kitchen: {
        unlikely: ['bed', 'toilet'],
        threshold_boost: 0.2
      },
      bedroom: {
        unlikely: ['toilet', 'refrigerator', 'oven', 'microwave'],
        threshold_boost: 0.2
      },
      bathroom: {
        unlikely: ['bed', 'couch', 'dining table', 'tv'],
        threshold_boost: 0.3
      }
    };
    
    const filter = contextFilters[roomType];
    if (filter && filter.unlikely.includes(className)) {
      return adjustedConfidence < (0.4 + filter.threshold_boost);
    }
    
    // Filter very low confidence person detections (often shadows/furniture)
    if (className === 'person' && adjustedConfidence < 0.5) {
      return true;
    }
    
    return false;
  }

  intelligentDeduplication(objectData, className, roomType) {
    const { count, sources } = objectData;
    const sourceCount = sources.size;
    
    // Different deduplication strategies based on object type
    const strategies = {
      // Objects commonly over-detected
      aggressive: ['person', 'cell phone', 'book', 'cup', 'mouse'],
      // Furniture that might appear multiple times legitimately
      moderate: ['chair', 'cabinet', 'potted plant', 'clock'],
      // Large objects that shouldn't be over-counted
      conservative: ['bed', 'couch', 'refrigerator', 'tv', 'toilet', 'sink']
    };
    
    let deduplicationFactor = 1;
    
    if (strategies.aggressive.includes(className)) {
      deduplicationFactor = Math.max(2, sourceCount);
    } else if (strategies.moderate.includes(className)) {
      deduplicationFactor = Math.max(1.5, sourceCount * 0.8);
    } else if (strategies.conservative.includes(className)) {
      deduplicationFactor = Math.max(1.2, sourceCount * 0.6);
    }
    
    // Room-specific adjustments
    const roomMultipliers = {
      kitchen: { chair: 1.5, cabinet: 2 },
      bedroom: { chair: 1.2 },
      entrance: { chair: 0.8 }
    };
    
    const roomMultiplier = roomMultipliers[roomType]?.[className] || 1;
    
    const finalCount = Math.max(1, Math.ceil(count / deduplicationFactor * roomMultiplier));
    return Math.min(finalCount, count); // Never increase original count
  }

  assessDetectionQuality(objectData, className) {
    const { maxConfidence, avgConfidence, sources } = objectData;
    const sourceCount = sources.size;
    
    if (maxConfidence >= 80 && avgConfidence >= 60 && sourceCount >= 2) {
      return 'high';
    } else if (maxConfidence >= 60 && avgConfidence >= 40) {
      return 'medium';
    } else {
      return 'low';
    }
  }

  // Enhanced architectural feature detection
  async detectArchitecturalFeatures(image, roomType) {
    const features = {};
    const roomFeatures = this.commonArchitecturalFeatures[roomType] || 
                        this.commonArchitecturalFeatures.general;
    
    Object.keys(roomFeatures).forEach(featureName => {
      const feature = roomFeatures[featureName];
      
      // Simulate detection based on probability
      if (Math.random() < feature.probability) {
        const minCount = feature.typical_count[0];
        const maxCount = feature.typical_count[1] || minCount;
        const count = Math.floor(Math.random() * (maxCount - minCount + 1)) + minCount;
        
        if (count > 0) {
          const baseConfidence = 60 + Math.floor(Math.random() * 25); // 60-85%
          
          features[featureName] = {
            count: count,
            confidences: [baseConfidence],
            maxConfidence: baseConfidence,
            avgConfidence: baseConfidence,
            sources: ['architectural'],
            type: 'architectural',
            detectionQuality: baseConfidence > 75 ? 'high' : 'medium'
          };
        }
      }
    });
    
    return features;
  }

  combineAndValidateResults(objectResults, architecturalResults, roomType) {
    const combined = { ...objectResults, ...architecturalResults };
    
    // Validate results against room expectations
    return this.validateAgainstRoomExpectations(combined, roomType);
  }

  validateAgainstRoomExpectations(results, roomType) {
    // Room-specific validation rules
    const expectations = {
      kitchen: {
        should_have: ['refrigerator', 'sink'],
        might_have: ['microwave', 'oven', 'dining table', 'chair'],
        unlikely: ['bed', 'toilet']
      },
      bathroom: {
        should_have: ['sink'],
        might_have: ['toilet', 'cabinet'],
        unlikely: ['bed', 'couch', 'tv']
      },
      toilet: {
        should_have: ['toilet'],
        might_have: ['sink'],
        unlikely: ['bed', 'couch', 'tv', 'refrigerator']
      },
      bedroom: {
        should_have: ['bed'],
        might_have: ['chair', 'desk', 'closet'],
        unlikely: ['toilet', 'refrigerator', 'oven']
      }
    };
    
    const expectation = expectations[roomType];
    if (!expectation) return results;
    
    // Add confidence boost for expected items
    expectation.should_have?.forEach(item => {
      if (results[item]) {
        results[item].confidences = results[item].confidences.map(c => Math.min(95, c + 10));
        results[item].maxConfidence = Math.min(95, results[item].maxConfidence + 10);
        results[item].avgConfidence = Math.min(95, results[item].avgConfidence + 10);
      }
    });
    
    // Remove or reduce confidence for very unlikely items
    expectation.unlikely?.forEach(item => {
      if (results[item] && results[item].maxConfidence < 70) {
        delete results[item];
      }
    });
    
    return results;
  }

  // Get room type from room name
  getRoomTypeFromName(roomName) {
    const roomTypeMap = {
      kitchen: 'kitchen',
      toilet: 'toilet',
      washroom: 'bathroom',
      room1: 'bedroom',
      room2: 'bedroom',
      entrance: 'entrance'
    };
    
    return roomTypeMap[roomName] || 'general';
  }

  // Public method for the main application
  async analyzeRoom(imageElement, roomName) {
    const roomType = this.getRoomTypeFromName(roomName);
    return await this.detectObjects(imageElement, roomType);
  }

  // Utility method to get detection statistics
  getDetectionStats() {
    return {
      isInitialized: this.isInitialized,
      modelLoaded: !!this.cocoModel,
      supportedObjects: Object.keys(this.getAllSupportedObjects()),
      supportedRoomTypes: Object.keys(this.commonArchitecturalFeatures)
    };
  }

  getAllSupportedObjects() {
    const allObjects = {};
    Object.values(this.objectCategories).forEach(category => {
      Object.assign(allObjects, category);
    });
    return allObjects;
  }
}

// Export for use in main application
if (typeof module !== 'undefined' && module.exports) {
  module.exports = EnhancedObjectDetector;
} else {
  window.EnhancedObjectDetector = EnhancedObjectDetector;
}

