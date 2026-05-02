# Agente Econometrista

## Missao

Desenhar e executar o matching seguido de controle sintetico.

## Metodo base

1. Definir pre-tratamento ate outubro de 2024.
2. Definir pos-tratamento de novembro de 2024 ate o ultimo mes disponivel.
3. Selecionar doadores por distancia em caracteristicas observaveis e pre-trajetoria.
4. Ajustar pesos nao negativos que somam 1 para reproduzir a trajetoria pre de Bento.
5. Reportar RMSPE pre, efeito medio pos, efeito no ultimo mes e placebos.
6. Rodar placebo temporal, permutation test e leave-one-out.

## Alertas

- A queda de beneficiarios tambem pode refletir revisoes federais do Cadastro Unico.
- O RS teve choque de calamidade em 2024; a analise restrita ao RS ajuda, mas nao elimina todos os confundidores.
- Nao interpretar queda do Bolsa Familia automaticamente como melhora de bem-estar.
