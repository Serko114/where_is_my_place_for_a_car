import matplotlib.pyplot as plt
import albumentations as albu
import numpy as np
import matplotlib.pyplot as plt

CLASSES = [
    "background",
    "ground",
    "no_car",
]
# --------------------------аугментация-------------------------------------------------


def get_validation_augmentation():
    # test_transform = [albu.Resize(height=256, width=256, p=1)]
    test_transform = [albu.LongestMaxSize(max_size=256, always_apply=True),
                      albu.PadIfNeeded(
                          min_height=256, min_width=256, border_mode=0, always_apply=True),
                      albu.CenterCrop(height=256, width=256, always_apply=True)]
    return albu.Compose(test_transform)
# --------------------------------------------------------------------------------------


def _colorize_mask(mask: np.ndarray, colors_imshow: dict):

    mask = mask.squeeze()
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    square_ratios = {}
    for cls_code, cls in enumerate(CLASSES):
        cls_mask = mask == cls_code
        square_ratios[cls] = cls_mask.sum() / cls_mask.size
        colored_mask += np.multiply.outer(cls_mask,
                                          colors_imshow[cls]).astype(np.uint8)
    # -----------------------------------------------------------
    # добавил коэффициент выход: отношение точек машин к точкам парковки:
    coeff = np.around(square_ratios['ground'] / square_ratios['no_car'], 4)
    print(coeff)
    # -----------------------------------------------------------
    return colored_mask, square_ratios, coeff


def reverse_normalize(img, mean, std):
    # Invert normalization
    img = img * np.array(std) + np.array(mean)
    return img


def visualize_predicts(img: np.ndarray, mask_gt: np.ndarray, mask_pred: np.ndarray, normalized=False):
    # размер img: H, W, CHANNEL
    # размер mask_gt, mask_pred: H, W, значения - range(len(CLASSES)
    # coeff = []
    _, axes = plt.subplots(1, 3, figsize=(10, 5))
    img = img.transpose(1, 2, 0)
    # print([square_ratios[cls] for cls in CLASSES])
    if normalized:
        # Reverse the normalization to get the unnormalized image
        img = reverse_normalize(img, mean=[0.485, 0.456, 0.406], std=[
                                0.229, 0.224, 0.225])
    axes[0].imshow(img)
    colors_imshow = {
        "background": np.array([0, 0, 0]),
        "ground": np.array([204, 153, 51]),
        "no_car": np.array([255, 96, 55]),
    }
    # axes[0].set_title('ngurnguir')

    mask_gt, square_ratios, coeff_real = _colorize_mask(mask_gt, colors_imshow)
    title = "Площади:\n" + f'COEFF_real: {str(coeff_real)}\n' + "\n".join(
        [f"{cls}: {square_ratios[cls]*100:.1f}%" for cls in CLASSES])
    axes[1].imshow(mask_gt, cmap="twilight")
    axes[1].set_title(f"GT маска\n" + title)

    mask_pred, square_ratios, coeff_pre = _colorize_mask(
        mask_pred, colors_imshow)
    title = "Площади:\n" + f'COEFF_pred: {str(coeff_pre)}\n' + "\n".join(
        [f"{cls}: {square_ratios[cls]*100:.1f}%" for cls in CLASSES])
    axes[2].imshow(mask_pred, cmap="twilight")
    axes[2].set_title(f"PRED маска\n" + title)

    plt.tight_layout()
    plt.show()
    return coeff_real, coeff_pre


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    # Осуществит стартовую нормализацию данных согласно своим значениям или готовым для imagenet
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)
# --------------------------------------------------------------------------
# ------------------------визуадизация--------------------------------------


def create_annotations_of_no_calsses_images(images_folder, annotations_folder):
    image_files = os.listdir(images_folder)
    for image_file in image_files:
        image_path = os.path.join(images_folder, image_file)
        annotation_path = os.path.join(
            annotations_folder, f"{os.path.splitext(image_file)[0]}.png")
        img = Image.open(image_path)
        width, height = img.size

        if not os.path.exists(annotation_path):
            img = Image.new('RGB', (width, height), color='black')
            img.save(annotation_path)


def _convert_multichannel2singlechannel(mc_mask: np.ndarray):
    """ Осуществляет перевод трехканальной маски (число каналов сколько классов) в трехканальное 
    изображение где будет расцветка как зададим в словаре colors_imshow для классов """

    colors_imshow = {
        "background": np.array([0, 0, 0]),
        "ground": np.array([255, 0, 0]),
        "no_car": np.array([0, 0, 255]),
    }

    sc_mask = np.zeros(
        (mc_mask[0].shape[0], mc_mask[0].shape[1], 3), dtype=np.uint8)
    square_ratios = {}

    for i, singlechannel_mask in enumerate(mc_mask):

        cls = CLASSES[i]
        singlechannel_mask = singlechannel_mask.squeeze()

        # Заодно осуществляет подсчет процента каждого класса (сумма пикселей на общее число)
        square_ratios[cls] = singlechannel_mask.sum() / singlechannel_mask.size

        sc_mask += np.multiply.outer(singlechannel_mask >
                                     0, colors_imshow[cls]).astype(np.uint8)

    title = "Площади: " + \
        "\n".join([f"{cls}: {square_ratios[cls]*100:.1f}%" for cls in CLASSES])
    return sc_mask, title


def visualize_multichennel_mask(img: np.ndarray, multichennel_mask: np.ndarray):
    """ Реализация демонстрации маски и самого изображения """
    # размер маски: H, W, CHANNEL
    _, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img)
    multichennel_mask = multichennel_mask.transpose(2, 0, 1)
    mask_to_show, title = _convert_multichannel2singlechannel(
        multichennel_mask)
    axes[1].imshow(mask_to_show)
    axes[1].set_title(title)

    plt.tight_layout()
    plt.show()
