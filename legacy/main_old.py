'''
images_path: pasta das imagens a serem processadas

save_folder_path: pasta onde as imagens serão salvas

equalization: método de equalização
  fn (função de equalização):
      rgb: equalização por canal individual do rgb
      rgb_cv2_gray: equalização por histograma em escala de cinza do opencv
      rgb_weighted_gray: equalização por histograma em escala de cinza ponderado
      hsv: equalização por canal individual do hsv (geralmente o v é o canal que mais importa)
      lab: equalização por canal individual do lab (geralmente o l é o canal que mais importa)
  args (argumentos da função de equalização):
      rgb: {}
      rgb_cv2_gray: {}
      rgb_weighted_gray: {'weights': [r, g, b]} (r, g, b = pesos dos canais rgb)
      hsv: {}
      lab: {}

channels: lista canais que serão equalizados
    ex: [0, 1, 2] (equaliza os 3 canais rgb)
        [2] (equal
        za apenas o canal azul)
        [0, 2] (equaliza os canais vermelho e azul)

align_margin: margem adicionada ao redor da moeda após o alinhamento

samples: pega um grupo aleatório de imagens (None = todas as imagens)

random_seed: semente aleatória para o método de amostragem
'''

from control_old import equalize_and_align

equalize_and_align(
    images_path='data/original',
    save_folder_path='data/original_results',
    equalization={
        'fn': 'rgb_weighted_gray',
        'args': {'weights': [1, 1, 1]}
    },
    channels=[0, 1, 2],
    show_histogram=False,
    align_margin=0.005,
    samples=None,
    random_seed=1011
)
