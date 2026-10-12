# Scientific Limitations & Research Roadmap

---

## 1. Documented Scientific & Engineering Limitations

1. **Spectral Range Constraint**: Current tiles are standard 3-channel RGB. Optical RGB cannot compute true Normalized Difference Water Index (NDWI) or Modified NDWI (MNDWI), which require Near-Infrared (NIR) Band 8 and Short-Wave Infrared (SWIR) Band 11.
2. **Cloud and Shadow Occlusion**: Deep cloud shadows over forested valleys can produce spectral signatures resembling clear water, inducing localized false positives.
3. **Physical Area Approximations**: Area calculations assume nominal 10m Ground Sampling Distance (GSD). Without embedded UTM CRS geospatial headers (e.g. GeoTIFF), area estimates must be treated as illustrative rather than certified geodetic surveys.
4. **Generalization Gap**: Performance across distinct geographic biomes (coastal deltas vs alpine glacial lakes) exhibits variance, highlighting the need for continental-scale training data.

---

## 2. Research Roadmap: GeoVision-IN Multi-Spectral Expansion

* **Phase 1: 4-Band & 12-Band Ingestion**: Ingest native Sentinel-2 SAFE / GeoTIFF formats to incorporate NIR (B8) and SWIR (B11/B12).
* **Phase 2: Physics-Informed Feature Engineering**: Concatenate native NDWI $(\frac{\text{Green} - \text{NIR}}{\text{Green} + \text{NIR}})$ and MNDWI $(\frac{\text{Green} - \text{SWIR}}{\text{Green} + \text{SWIR}})$ channels into the model input tensor.
* **Phase 3: Continuous Water-Fraction Regression**: Extend from hard binary thresholding to continuous sub-pixel water fraction regression.
* **Phase 4: Indian River Basin & Monsoon Benchmark**: Curate annotated tiles across the Ganges-Brahmaputra and Godavari deltas across pre-monsoon and post-monsoon seasons to evaluate seasonal flood detection.
