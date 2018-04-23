import pandas as pd
import shapely.ops
import shapely.geometry
import rasterio.features
import os
import numpy as np

# Parameters
MIN_POLYGON_AREA = 30

def mask_to_poly(mask, min_polygon_area_th=MIN_POLYGON_AREA):
    mask = (mask > 0.5).astype(np.uint8)
    shapes = rasterio.features.shapes(mask.astype(np.int16), mask > 0)
    for elem in shapes:
        print(elem)
    poly_list = []
    mp = shapely.ops.cascaded_union(
        shapely.geometry.MultiPolygon([
            shapely.geometry.shape(shape)
            for shape, value in shapes
        ]))

    if isinstance(mp, shapely.geometry.Polygon):
        df = pd.DataFrame({
            'area_size': [mp.area],
            'poly': [mp],
        })
    else:
        df = pd.DataFrame({
            'area_size': [p.area for p in mp],
            'poly': [p for p in mp],
        })

    df = df[df.area_size > min_polygon_area_th].sort_values(
        by='area_size', ascending=False)
    df.loc[:, 'wkt'] = df.poly.apply(lambda x: shapely.wkt.dumps(
        x, rounding_precision=0))
    df.loc[:, 'bid'] = list(range(1, len(df) + 1))
    df.loc[:, 'area_ratio'] = df.area_size / df.area_size.max()
    return df

def _internal_pred_to_poly_file_test(y_pred,
                                     min_th=MIN_POLYGON_AREA):
    """
    Write out test poly
    """
    #prefix = area_id_to_prefix(area_id)

    # Load test imagelist
    fn_test = "output.csv" #FMT_TEST_IMAGELIST_PATH.format(prefix=prefix)
    # df_test = pd.read_csv(fn_test, index_col='ImageId')
    df_test = pd.read_csv(fn_test)

    # Make parent directory
    fn_out = "fjroutput" #FMT_TESTPOLY_PATH.format(prefix)
    #if not Path(fn_out).parent.exists():
    #    Path(fn_out).parent.mkdir(parents=True)

    # Ensemble individual models and write out output files
    with open(fn_out, 'w') as f:
        f.write("ImageId,BuildingId,PolygonWKT_Pix,Confidence\n")
        # for idx, image_id in enumerate(df_test.index.tolist()):
        for idx in range(381):
            df_poly = mask_to_poly(y_pred[idx], min_polygon_area_th=min_th)
            if len(df_poly) > 0:
                for i, row in df_poly.iterrows():
                    line = "{},{},\"{}\",{:.6f}\n".format(
                        idx,
                        row.bid,
                        row.wkt,
                        row.area_ratio)
                    line = _remove_interiors(line)
                    f.write(line)
            else:
                f.write("{},{},{},0\n".format(
                    idx,
                    -1,
                    "POLYGON EMPTY"))



if __name__ == '__main__':
    # 1. read np array as y_pred
    MODEL = 'AOI_3_Paris'
    IMG_LIST_FN = os.environ['PWD'] +  '/' + MODEL + '_test_ImageId.csv'
    NUMPY_FILE  = os.environ['PWD'] +  '/' + MODEL + '_poly.npy'
    PNG_DIR     = os.environ['PWD'] +  '/' + MODEL + '_test_png_v17_fjrout'
    image_array = np.load(NUMPY_FILE)
    _internal_pred_to_poly_file_test(image_array)

