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

  3.1	Univarijantna analiza podataka
    3.1.1	Kategorijske promenljive
    3.1.2	Numeričke Promenljive
5.	Čišćenje podataka	
  4.1	Uklanjanje nepravilnih vrednosti
  4.2	Rešavanje Nedostajućih Vrednosti
  4.3	Rešavanje Outliera (Ekstremnih Vrednosti)
6.	Bivarijantna analiza podataka
  5.1	Odnos sa promenljivom ,,satisfaction’’	
  5.2	Redukcija Podataka	
7.	Feature Engineering	
  6.1	Stvaranje novih promenljivih	
  6.2	Transformacija podataka (Kodiranje Kategorijskih podataka)	
8.	Modelovanje	
  7.1	Logistička regresija	
  7.2	Decision Tree	
  7.3	Random Forest	
  7.4	XGBoost	
9.	Poređenje modela i finalna evaluacija	
  8.1	ROC krive za sve modele	
  8.2	Konfuziona matrica za najbolji model	
  8.3	Feature Importance analiza	
10.	Zaključak	


