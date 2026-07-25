# Stage 0 — Data Preparation Summary

- Raw products scanned: 147702
- Products with an English title: 122734
- Products with a main_image_id: 122274
- Products in the home/furniture/household category set: 12249
- Products after joining with image metadata: 12208
- Products after de-duplicating product_id: 12208
- **Final catalog size: 2000**
- Unique product types in final catalog: 177
- Missing title rate: 0.0000%
- Missing image rate: 0.0000%
- Average product_text length (chars): 690.5
- Median product_text length (chars): 690.5

## product_type distribution (top 20)

| product_type | count |
|---|---:|
| BED | 30 |
| DESK | 30 |
| CABINET | 30 |
| CHAIR | 30 |
| DRINKING_CUP | 29 |
| STORAGE_HOOK | 29 |
| BENCH | 29 |
| FOOD_SERVICE_SUPPLY | 29 |
| HOME_LIGHTING_AND_LAMPS | 29 |
| SWATCH | 29 |
| DRESSER | 29 |
| BED_FRAME | 29 |
| JANITORIAL_SUPPLY | 29 |
| VASE | 29 |
| BASKET | 29 |
| CLOTHES_HANGER | 29 |
| CLOTHES_RACK | 29 |
| BEAN_BAG_CHAIR | 29 |
| STORAGE_BOX | 29 |
| FLATWARE | 29 |

## 10 example final products

| product_id | title | product_type | category_path |
|---|---|---|---|
| B07B4Z7MRL|amazon.com | Amazon Brand – Stone & Beam Chase Casual Wood Underbed Storage, 80", N | STORAGE_BOX | Storage & Organization/Clothing & Closet Storage/Under-Bed Storage |
| B06X93TB4S|amazon.in | Amazon Brand - Solimo 24 Piece Stainless Steel Cutlery Set, Waves (Con | FLATWARE | Kitchen & Dining/Tableware/Cutlery & Flatware/Cutlery & Flatware Sets/Mixed Cutlery & Flatware Sets |
| B07FBNZW5W|amazon.com | AmazonBasics Ultra-Soft Flat Sheet - Queen | FLAT_SHEET | Bedding/Sheets & Pillowcases/Flat Sheets |
| B07K6MBGYH|amazon.com | AmazonBasics Heavy Duty Non-Slip Bed Frame with Steel Slats, Easy Asse | BED_FRAME | Furniture/Bedroom Furniture/Beds, Frames & Bases/Bed Frames |
| B07XPT5K1V|amazon.com | AmazonBasics 6-Tier Hanging Shelf Closet Storage Organizer with Remova | SHELF | Storage & Organization/Clothing & Closet Storage/Closet Rods & Shelves/Closet Shelves |
| B07C3SH85X|amazon.com.au | AmazonBasics Ultra-Soft Bedding Singles, 100% Cotton, Dove Grey, Twin | FLAT_SHEET | Bedding & Linen/Sheets & Pillowcases |
| B07RYBB1L5|amazon.com | AmazonBasics 3-Tier Metal Rolling Storage Cart and Organizer, Turquois | TEACHING_EQUIPMENT | Office Furniture & Lighting/Carts & Stands/Utility Carts |
| B07LC6DLJW|amazon.in | Amazon Brand - Solimo Alpha Engineered Wood 3-Door Wardrobe (Oak Finis | CABINET | Furniture/Bedroom Furniture/Bedroom Wardrobes |
| B07YG8QZ45|amazon.com.au | AmazonBasics Oversize Outdoor Market Patio Umbrella with Base - 15 x 6 | UMBRELLA | Patio Furniture & Accessories/Umbrellas & Shade |
| B07GL54DTV|amazon.in | Amazon Brand - Solimo Round Lunch Box with Fork and Spoon, 1.1 Litre,  | MEAL_HOLDER | Kitchen & Dining/Kitchen Storage & Containers/Lunch Boxes |

## Notes

- Only the ABO small-image archive (largest axis <= 256px) and the listings/image metadata archives were used.
- The full-resolution image archive, 360-degree image archive, and 3D model archive were **not** used, as required for the base project.
