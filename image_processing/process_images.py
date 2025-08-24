import concurrent.futures
import os
import ray
import requests
from huggingface_hub import HfFileSystem


def fetch_image_original(row):
    try:
        response = requests.get(row["url"], timeout=5)
        if response.status_code == 200:
            row["image_bytes"] = response.content
            row["success"] = True
            row["error"] = None
        else:
            row["image_bytes"] = None
            row["success"] = False
            row["error"] = f"Status code: {response.status_code}"
        return row
    except Exception as e:
        row["image_bytes"] = None
        row["success"] = False
        row["error"] = str(e)
        return row


def fetch_image(url):
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            image_bytes = response.content
            success = True
            error = None
        else:
            image_bytes = None
            success = False
            error = f"Status code: {response.status_code}"
    except Exception as e:
        image_bytes = None
        success = False
        error = str(e)

    return image_bytes, success, error


def fetch_images_batch_threaded(batch):
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        results = list(executor.map(fetch_image, batch["url"]))
    batch["image_bytes"] = [result[0] for result in results]
    batch["success"] = [result[1] for result in results]
    batch["error"] = [result[2] for result in results]
    return batch


dataset = ray.data.read_parquet(
    "hf://datasets/laion/relaion2B-en-research-safe/",
    file_extensions=["parquet"],
    filesystem=HfFileSystem(token=os.environ["HF_TOKEN"]),
    ray_remote_args={"memory": 10*10**9}
)
dataset = dataset.repartition(target_num_rows_per_block=5000)
dataset = dataset.map_batches(
    fetch_images_batch_threaded,
    batch_size=1000,
    # ray_remote_args_fn=lambda: {
    #     "memory": 2 * 10**9
    # },
)
dataset = dataset.materialize()

"""
urls = [
    'https://lid.zoocdn.com/354/255/1c262cde0e91356edf2f1ddad3e92ae8ccf9dd98.jpg',
    'http://tse2.mm.bing.net/th?id=OIP.Gqs8tMg9NcakYZVveTRSfQEsEs',
    'https://shop.foleyfoodandwinesociety.com/assets/images/products/pictures/21021-560.png',
    'https://d1ea30dbll17d8.cloudfront.net/12/1/images/catalog/i/xl_137503-P7010009.jpg',
    'http://rlv.zcache.com/pomeranian_christmas_card_santa_and_bears-r500df5b7d0f94c258ca220c7110273bf_xvuak_8byvr_152.jpg',
    'https://reisetopia.de/wp-content/uploads/2018/04/Swiss-First-Class-Internet-Voucher-1024x576.jpg',
    'http://tse3.mm.bing.net/th?id=OIP.g1eGMQVWD1zBOqlSW6tG9wHaHf',
    'https://tap2.fkimg.com/media/vr-splice-j/01/41/0b/04.jpg',
    'https://i.pinimg.com/736x/3f/e5/32/3fe532c33b9babe6ed6df71a137891d0.jpg',
    'https://s3-us-west-2.amazonaws.com/tabs.web.media/c/7/c7vf/c7vf-square-175.jpg',
    'https://secure.img.wfrcdn.com/lf/43/hash/6189/7540185/1/26%2BBottle%2BSingle%2BZone%2BBuilt-In%2BWine%2BRefrigerator.jpg',
    'https://storage.googleapis.com/idx-photos-gs.ihouseprd.com/OR-RMLS/17075635/1x/000.jpg',
    'http://cdn-w.v12soft.com/photos/8PtebOg/9122296/fszBM_800600.jpg',
    'https://blogs.gnome.org/aklapper/files/2011/03/sponsored-gnome-badge-shadow.png',
    'https://laclothing.co.uk/wp-content/uploads/prod_dp37_88695-400x533.jpg',
    'https://onokovtsy.diamondelectric.ru/images/2957/2956669/small_kofemashina_philips_ep503510_lattego_series_5000_1.jpg',
    'https://www.enigmasoftware.com/wp-content/themes/default/images/pages/download/spyhunter/chrome/2.jpg',
    'https://media.vanityfair.com/photos/54cab13f674871890b5748f9/master/w_320%2Cc_limit/image.jpg',
    'http://ih1.redbubble.net/image.16549013.9491/flat,220x200,075,t.jpg',
    'https://www.cstatic-images.com/stock/400x300/265593.jpg',
    'https://image.spreadshirtmedia.net/image-server/v1/products/T1437A737PA4399PT17X218Y32D174571818FS798/views/1,width=500,height=500,appearanceId=737/diagramme-t-shirt-manches-longues-henley.jpg',
    'https://www.picclickimg.com/d/l400/pict/371869372230_/South-Carolina-Camper.jpg',
    'https://www.amysbakehouse.com.au/wp-content/uploads/2015/08/30thBirthdayCake5-300x300.jpg',
    'https://images-eu.ssl-images-amazon.com/images/I/51uUj7YpvAL._AC_UL160_SR102,160_.jpg',
    'https://i.ytimg.com/vi/Bas_HPoslzY/mqdefault.jpg',
    'https://aviewfrommyseat.com/medium/anonymous-20150912164947.jpg',
    'https://auto.cdn-rivamedia.com/photos/annonce/bigcriteo/mercedes-classe-c-180-d-122ch-amg-line-9g-tronic-117174350.jpg',
    'https://ae01.alicdn.com/kf/HTB1BSAEjYSYBuNjSspiq6xNzpXaS/Custom-photo-3d-wallpaper-cloth-Motorcycle-retro-nostalgic-living-room-Home-decor-3d-wall-murals-wallpaper.jpg',
    'http://booklikes.com/photo/max/220/330/upload/books/b/5/b50d9f8dd976cb135363e03ee6f279fd.jpg',
    'https://images.snapwi.re/d571/5a675db017312ea0328b456f.w800.jpg',
    'https://m.smedata.sk/api-media/media/image/sme/0/43/4366330/4366330_600x400.jpeg?rev=3',
    'https://d31l02nbp0owar.cloudfront.net/m/t/15589/15579430/a-0005.jpg',
    'https://img.shopstyle-cdn.com/pim/b2/c7/b2c7ad1e66c19e0aef0fad8464580abd_xlarge.jpg',
    'https://www.fineartstorehouse.com/t/629/greater-yellowlegs-reflects-13091205.jpg.webp',
    'https://media.ticmate.com/resources/ticmate_live/upload/leeds_united_team_logo.jpg',
    'http://d3d71ba2asa5oz.cloudfront.net/62000804/images/by0006_2.jpg',
    'https://fscomps.fotosearch.com/compc/CSP/CSP990/seamless-light-oak-square-parquet-panel-clipart__k10885495.jpg',
    'https://i.pinimg.com/236x/29/86/6b/29866b98be3977e1f8f7c58c27a1607d.jpg'
]
"""