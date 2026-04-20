from __future__ import annotations

from core.predict import ModelPaths, predict_flag, predict_freight


def main() -> None:
    paths = ModelPaths()

    freight = predict_freight(1500, paths)
    print("freight prediction:", freight)

    flag = predict_flag(
        {
            "invoice_quantity": 10,
            "invoice_dollars": 1500,
            "freight_invoiced": 25,
            "total_item_quantity": 10,
            "total_item_dollars": 1498,
        },
        paths,
    )
    print("flag prediction:", flag)


if __name__ == "__main__":
    main()

