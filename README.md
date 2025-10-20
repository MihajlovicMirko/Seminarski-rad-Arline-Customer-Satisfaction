# Seminarski-rad-Airline-Customer-Satisfaction-
U ovom seminarskom radu analiziraćemo skup podataka “Airline Customer Satisfaction” (stvorio ju je Ramin Huseyn), i napraviti model za predikciju promenljive satisfaction (zadovoljstvo) - analiziraćemo koliko određene informacije prikupljene iz skupa utiču na celokupno uživanje putnika tokom vožnje.

## Fajlovi

- **Airline_Customer_Satisfaction.url**: Link do strane gde smo našli ovaj skup podataka.  
- **Airline_Customer_Satisfaction.csv**: Skup podataka.  
- **README.md**: Fajl na kome je napisan ovaj opis.  
- **Seminarski rad - kod.ipynb**: Celokupan kod izvršavanja zadataka u našem radu.
- **Seminarski rad - uvod u nauku o podacima.doc**: Dokument sa radom.
- **clean_airline_data.csv**:Skup podataka nakon transformacije (Skaliranja, Label, One Hot i Ordinal enkodiranja).


## Skup Podataka
Ovaj skup podataka sadrži informacije o kvalitetu usluge na avionu, opis samog putnika kao i opis samog puta (dužina puta)
Imamo sledeće kolone:

### Kategorijske promenljive
- **Satisfaction**: Zadovoljstvo putnika.  
- **Customer Type**: Lojalan ili nelojalan.  
- **Type Of Travel**: Poslovni put ili zadovoljstvo.  
- **Class**: Eco, Eco plus, Business.

### Numeričke promenljive
- **Age**: Starost putnika.  
- **Flight Distance**: dužina puta.
- **Departure Delay in Minutes** koliko kasni polazak aviona u minutima.
- **Arrival  Delay in Minutes** koliko kasni dolazak aviona u minutima.

### Ocene  
- **Seat comfort** udobnost sedišta (1-5) 
- **Departure/Arrival time convenient** Da li je pogodno vreme dolaska/odlaska (1-5)
- **Food and drink** Hrana i piće (1-5)
- **Inflight wifi service** kvalitet wifi-a (1-5)
- **Gate location** Da li odgovara lokacija izlaska (1-5)
- **Inflight entertainment** Zabava na avionu (1-5)
- **Online support booking** Online podrška (1-5)
- **Ease of Online** Lakoća online rezervacije (1-5)
- **On-board service** Servis u letu (1-5)
- **Leg room service** Prostor za noge (1-5)
- **Baggage handling** Rukovanje prtljagom (1-5)
- **Checkin service** Servis za čekiranje (1-5)
- **Cleanliness** Čistoća (1-5)
- **Online boarding** Zadovoljstvo online ukrcavanjem  (1-5)

## Sadržaj rada
1.	Uvod
2.	Učitavanje podataka
3.	Eksploratorna analiza podataka (EDA)
4.	Čišćenje podataka	- Uklanjanje nepravilnih vrednosti, Rešavanje Nedostajućih Vrednosti, Rešavanje Outliera (Ekstremnih Vrednosti)
5.	Bivarijantna analiza podataka - Odnos sa promenljivom ,,satisfaction’’ i redukcija
6.	Feature Engineering	
7.	Modelovanje	
8.	Poređenje modela i finalna evaluacija		
9.	Zaključak 


