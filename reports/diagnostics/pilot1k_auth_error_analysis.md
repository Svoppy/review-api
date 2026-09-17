# Error Analysis Report

- Task: `authenticity`
- Model A: `multitask`
- Model B: `single_task_auth`

## Slice-level error-rate comparison

| Slice | Support | Model A error rate | Model B error rate | A-B gap |
|---|---:|---:|---:|---:|
| `language:tr` | 100 | 0.0600 | 0.3600 | -0.3000 |
| `language:es` | 100 | 0.0300 | 0.2700 | -0.2400 |
| `length_bucket:long` | 191 | 0.0733 | 0.2984 | -0.2251 |
| `language:en` | 83 | 0.1205 | 0.3373 | -0.2169 |
| `language:ru` | 115 | 0.0522 | 0.2174 | -0.1652 |
| `language:it` | 90 | 0.1000 | 0.2444 | -0.1444 |
| `domain:hospitality` | 1000 | 0.0990 | 0.2180 | -0.1190 |
| `source:maide_up` | 1000 | 0.0990 | 0.2180 | -0.1190 |
| `length_bucket:medium` | 485 | 0.1113 | 0.2247 | -0.1134 |
| `length_bucket:short` | 324 | 0.0957 | 0.1605 | -0.0648 |
| `language:ro` | 100 | 0.0900 | 0.1400 | -0.0500 |
| `language:de` | 110 | 0.1455 | 0.1909 | -0.0455 |
| `language:zh` | 115 | 0.0957 | 0.1304 | -0.0348 |
| `language:fr` | 85 | 0.0706 | 0.0941 | -0.0235 |
| `language:ko` | 102 | 0.2255 | 0.2157 | 0.0098 |

## Examples where `multitask` is correct and `single_task_auth` is wrong

- source=`maide_up` language=`en` domain=`hospitality` gold=`authentic` a=`authentic` b=`fake` text=`Pros: cleanliness is always up to my standard. Cons: I frequently stay in this hotel, I love everything about it, the location, the food and the kind staff.`
- source=`maide_up` language=`es` domain=`hospitality` gold=`authentic` a=`authentic` b=`fake` text=`Pros: estaba limpio, bien ubicado y en buen estado, el personal muy amable. Cons: Era extremadamente pequeña. Dos personas en una habitación doble, paradas y dos Carry One no cabíamos. Es muy literal que las habitaciones son muy muy muy peq`
- source=`maide_up` language=`it` domain=`hospitality` gold=`authentic` a=`authentic` b=`fake` text=`Pros: Buona la colazione. Cons: La stanza dava su una strada rumorosissima. La luce del letto non funzionava. Il letto tremava e non avevo una scatola dove mettere le mie cose come gli altri ospiti. La doccia aveva un pezzo staccato dal mur`
- source=`maide_up` language=`tr` domain=`hospitality` gold=`authentic` a=`authentic` b=`fake` text=`Pros: konumu mukemmel, kisa sureli gidenler icin harika bir konumda. her yere yurume mesafesi Cons: odalar biraz fazla kucuk ve ses yalitimi kotu`
- source=`maide_up` language=`en` domain=`hospitality` gold=`authentic` a=`authentic` b=`fake` text=`Pros: Convenient location, unique design of the hotel, beautiful decoration of exterior and landscape design, nice interior, nice smell inside hotel, big and clean room with big windows, everything inside room works fine. Cons: didn't notic`
- source=`maide_up` language=`tr` domain=`hospitality` gold=`authentic` a=`authentic` b=`fake` text=`Pros: oda temiz ve konforluydu. kahvalti iyiydi. Cons: klima maalesef sogutmuyordu. balayi odasi yan binaya bakiyordu, bu sebeple neredeyse hic perdeleri acamadik. otelde alkollu hicbir icecek yoktu.`
- source=`maide_up` language=`ro` domain=`hospitality` gold=`authentic` a=`authentic` b=`fake` text=`Pros: Micul dejun a fost foarte bun. Camere curate si cu tot ce e necesar unei sederi scurte. Este low budget ca si finisaje dar totul e foarte bine intretinut. Pozitionat excelent aproape de centru. Cons: Camera de 3 persoane a fost de fap`
- source=`maide_up` language=`en` domain=`hospitality` gold=`authentic` a=`authentic` b=`fake` text=`Pros: Very nice, boutique hotel, walking distance to old town, supermarket next door, walking distance restaurant, MacDonalds next door, good breakfast. Myself and my daughter really enjoyed. Many thanks to the reception staff, very helpful`
- source=`maide_up` language=`es` domain=`hospitality` gold=`authentic` a=`authentic` b=`fake` text=`Pros: El hotel es sencillo pero tiene todo lo que se necesita. La ubicación es muy buena rodeador de restaurantes, cafeterías y muy cerca del Louvre Cons: Nada en concreto`
- source=`maide_up` language=`de` domain=`hospitality` gold=`authentic` a=`authentic` b=`fake` text=`Pros: Das Ambiente, die Lobby, die Auswahl beim Frühstück, der Spa-Bereich, die Freundlichkeit einiger Angestellter an der Rezeption und beim Frühstück. Cons: Am ersten Morgen konnten wir unser Zimmer nicht verlassen, weil der gesamte Türra`

## Examples where `single_task_auth` is correct and `multitask` is wrong

- source=`maide_up` language=`ru` domain=`hospitality` gold=`fake` a=`authentic` b=`fake` text=`Pros: Шикарный вид на город, удобное расположение Cons: Не очень чистый номер, в ванной комнате обнаружили плесень, не слишком приятный персонал`
- source=`maide_up` language=`ro` domain=`hospitality` gold=`authentic` a=`fake` b=`authentic` text=`Pros: Locatie buna, personal exceptional, servicii de calitate, locatie curata. Am fost foarte multumita si recomand!`
- source=`maide_up` language=`ko` domain=`hospitality` gold=`authentic` a=`fake` b=`authentic` text=`Pros: 직원들이 매우 친절함 가격이 저렴함 강남 한복판에서 놀고 싶다면 괜찮은 장소 전자렌지가 있음 Cons: 동네가 너무 놀자판이라 조용하고 편안한 휴식을 원하면 비추 새벽부터 오토바이, 자동차 굉음이 매우 심함 에어컨, 화장실 환풍기가 소음공해 수준 아침에 어디 선가 들어오는 담배냄새`
- source=`maide_up` language=`de` domain=`hospitality` gold=`authentic` a=`fake` b=`authentic` text=`Pros: moderne Räume Cons: Personal war extrem unfreundlich. Ob Rezeption oder Spa`
- source=`maide_up` language=`ro` domain=`hospitality` gold=`fake` a=`authentic` b=`fake` text=`Pros: Personal minunat, facilități excelente, super curat Cons: Zgomotul de la etajul de jos se aude în cameră, dar nu este foarte deranjant`
- source=`maide_up` language=`it` domain=`hospitality` gold=`fake` a=`authentic` b=`fake` text=`Pros: Posizione centralissima a un passo da tutte le attrazioni di Parigi. Lo staff è molto attento e la colazione più che decente. Stanza pulita e comoda. Cons: Nessuna nota negativa da segnalare`
- source=`maide_up` language=`zh` domain=`hospitality` gold=`authentic` a=`fake` b=`authentic` text=`Pros: 因为突发情况我在最后一刻预定了房间并需要立刻存放行李。屋主展示出高水准的服务品质和应对紧急情况的灵活性。给我全面的入住须知和旅行向导，并提醒我提前预定机场大巴的车票。 酒店高品质的设施精致简约，处处彰显屋主的卓越的设计品味。`
- source=`maide_up` language=`zh` domain=`hospitality` gold=`fake` a=`authentic` b=`fake` text=`Pros: 正在羅馬市中心，資源可用，平靠很寻～ Cons: 上網速度有些慢`
- source=`maide_up` language=`it` domain=`hospitality` gold=`fake` a=`authentic` b=`fake` text=`Pros: Posizione conveniente prossima alla metropolitana Cons: Camera sporca, cibo a colazione di scarsa qualità, personale poco disponibile ed educato`
- source=`maide_up` language=`fr` domain=`hospitality` gold=`authentic` a=`fake` b=`authentic` text=`Pros: l accueil et la disponibilité de tout le personnel Très bon petit déjeuner avec des produits de qualité Emplacement géographique très important pour se déplacer dans Paris Stationnement facile et restaurants abondant dans le quartier`

