import SimpleITK as sitk


def convert_to_nifti(input_dicom_directory: str, output_file: str) -> None:
    """
    This function performs NIfTI conversion of a DICOM image series.

    Args:
        input_dicom_directory (str): The path to a DICOM series.
        output_file (str): The output NIfTI file.
    """
    print("Starting DICOM to NIfTI conversion using ITK...")
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(input_dicom_directory)
    if not series_IDs:
        raise ValueError(
            'ERROR: given directory "'
            + input_dicom_directory
            + '" does not contain a valid DICOM series.'
        )
    if len(series_IDs) > 1:
        print(
            'WARNING: given directory "'
            + input_dicom_directory
            + '" contains more than 1 DICOM series; only the first one, "'
            + series_IDs[0]
            + '" will be converted.',
            flush=True,
        )
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
        input_dicom_directory, series_IDs[0]
    )

    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)

    series_reader.MetaDataDictionaryArrayUpdateOn()
    series_reader.LoadPrivateTagsOn()
    image_dicom = series_reader.Execute()

    sitk.WriteImage(image_dicom, output_file)
